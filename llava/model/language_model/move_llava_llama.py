#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
import numpy as np

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..multimodal_encoder.mixture_vision_models import init_vision_teachers
from ..multimodal_encoder.moe_lora import MoE_LoRA_CLIP, MoECLIPEncoderLayer, MoECLIPEncoder, MoECLIPVisionTransformer
from PIL import Image


class MoVELlavaLlamaConfig(LlamaConfig):
    model_type = "move_llava_llama"
    def __init__(self,
                 kd_mode=True,
                 moe_encoder=True,
                 moe_encoder_dense=False,
                 moe_encoder_lora_rank=32,
                 moe_encoder_lora_alpha=1,
                 moe_encoder_num_experts=3,
                 **kwargs):
        self.moe = dict(
            moe_encoder=moe_encoder,
            moe_encoder_dense=moe_encoder_dense,
            moe_encoder_lora_rank=moe_encoder_lora_rank,
            moe_encoder_lora_alpha=moe_encoder_lora_alpha,
            moe_encoder_num_experts=moe_encoder_num_experts,
        )
        self.kd = dict(
            kd_mode = kd_mode
        )

        super(MoVELlavaLlamaConfig, self).__init__(**kwargs)


class MoVELlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = MoVELlavaLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(MoVELlavaLlamaModel, self).__init__(config)


class MoVELlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = MoVELlavaLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MoVELlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.generate_mode = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def initialize_teachers(self, model_args, training_args, data_args):
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.image_process = self.model.get_vision_tower().image_processor

        self.config.kd['kd_mode'] = self.training_args.kd

        if self.model_args.encoder_teachers:
            self.vision_teachers = init_vision_teachers(self.model_args)
            for name, teacher in self.vision_teachers.items():
                self.vision_teachers[name] = teacher.cuda().to(torch.bfloat16)
            self.num_teachers = len(self.vision_teachers.keys())
            self.teachers_list = sorted(self.vision_teachers.keys()) 
            self.gelu = torch.nn.GELU()
            for name in self.vision_teachers.keys():
                if name == 'SAM':
                    self.sam_adapter_1 = nn.Linear(256, 1024).cuda().to(torch.bfloat16)
                    self.sam_adapter_2 = nn.Linear(4096, 576).cuda().to(torch.bfloat16)
                elif name == 'EVA':
                    self.eva_adapter_1 = nn.Linear(1024, 1024).cuda().to(torch.bfloat16)
                    self.eva_adapter_2 = nn.Linear(4096, 576).cuda().to(torch.bfloat16)
                elif name == 'Pix2Struct':
                    self.pix2struct_adapter_1 = nn.Linear(1536, 1024).cuda().to(torch.bfloat16)
                    self.pix2struct_adapter_2 = nn.Linear(1024, 576).cuda().to(torch.bfloat16)
                elif name == 'ConvNeXt':
                    self.convnext_adapter_1 = nn.Linear(3072, 1024).cuda().to(torch.bfloat16)
                    self.convnext_adapter_2 = nn.Linear(1024, 576).cuda().to(torch.bfloat16)
        else:
            self.vision_teachers = None
        
        self.config.moe['moe_encoder'] = self.model_args.moe_encoder
        if self.model_args.moe_encoder:
            self.config.moe['moe_encoder_dense'] = self.model_args.moe_encoder_dense
            self.config.moe['moe_encoder_lora_rank'] = self.model_args.moe_encoder_lora_rank
            self.config.moe['moe_encoder_lora_alpha'] = self.model_args.moe_encoder_lora_alpha
            self.config.moe['moe_encoder_num_experts'] = self.model_args.moe_encoder_num_experts
            # replace origin_model with moe_model
            origin_model = self.model.get_vision_tower().vision_tower.vision_model
            moe_model = MoECLIPVisionTransformer(origin_model.config)
            moe_model.load_state_dict(self.model.get_vision_tower().vision_tower.vision_model.state_dict())
            self.model.get_vision_tower().vision_tower.vision_model = moe_model

            # replace origin_encoder with moe_encoder
            origin_encoder = self.model.get_vision_tower().vision_tower.vision_model.encoder
            moe_encoder = MoECLIPEncoder(origin_encoder.config)
            moe_encoder.load_state_dict(self.model.get_vision_tower().vision_tower.vision_model.encoder.state_dict())
            self.model.get_vision_tower().vision_tower.vision_model.encoder = moe_encoder

            # replace origin_layer with moe_layer
            for i, layer in enumerate(self.model.get_vision_tower().vision_tower.vision_model.encoder.layers):
                moe_layer = MoECLIPEncoderLayer(self.model.get_vision_tower().vision_tower.vision_model.encoder.config)
                moe_layer.load_state_dict(layer.state_dict())
                self.model.get_vision_tower().vision_tower.vision_model.encoder.layers[i] = moe_layer

            # replace origin_mlp with moe_mlp
            for i, layer in enumerate(self.model.get_vision_tower().vision_tower.vision_model.encoder.layers):
                origin_mlp = layer.mlp
                self.model.get_vision_tower().vision_tower.vision_model.encoder.layers[i].mlp = \
                    MoE_LoRA_CLIP(
                        args=self.model_args,
                        lora_rank=self.model_args.moe_encoder_lora_rank,
                        lora_alpha=self.model_args.moe_encoder_lora_alpha,
                        num_experts=self.model_args.moe_encoder_num_experts,
                        dense_moe=self.model_args.moe_encoder_dense,
                        original_module=origin_mlp
                    )
        self.model.get_vision_tower().requires_grad_(False)
            
        

    def process_images(self, images,image_process):
       if self.data_args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            images = [
                        expand2square(images[i], tuple(int(x * 255) for x in image_process.image_mean))
                        if not isinstance(images[i], torch.Tensor)  
                        else images[i]  
                        for i in range(len(images))
                    ]
       image_processed = torch.stack([image_process.preprocess(images[i], 
                                            return_tensors='pt')['pixel_values'][0]
                                            for i in range(len(images))], dim=0)
       return image_processed.to(self.model.dtype)

    def _iter_moe_layers(self):
        if not self.config.moe['moe_encoder']:
            return []
        layers = self.model.get_vision_tower().vision_tower.vision_model.encoder.layers
        return [layer.mlp for layer in layers if isinstance(layer.mlp, MoE_LoRA_CLIP)]

    def _set_moe_forward_mode(self, mode="router", forced_expert_idx=None, router_train_only=False):
        for moe_mlp in self._iter_moe_layers():
            moe_mlp.forward_mode = mode
            moe_mlp.forced_expert_idx = forced_expert_idx
            moe_mlp.router_train_only = router_train_only

    def _compute_teacher_tokens(self, images):
        teacher_tokens = {}
        teacher_weight = {}
        token_weight = None
        cls_token = None
        for name, vision_teacher in self.vision_teachers.items():
            with torch.no_grad():
                vision_teacher.eval()
                image_processor = vision_teacher.image_processor
                image_processed = self.process_images(images, image_processor)
                if name == 'CLIP':
                    token, teacher_out = vision_teacher(image_processed)
                    attentions = teacher_out.attentions[self.model_args.mm_vision_select_layer].mean(1)
                    token_weight = torch.softmax(attentions[:, 0, 1:], dim=-1)
                    cls_token = teacher_out.hidden_states[self.model_args.mm_vision_select_layer][:, 0, :]
                else:
                    token = vision_teacher(image_processed)
                if isinstance(token, tuple):
                    token = token[0]

            if name == 'SAM':
                token = token.view(token.size(0), token.size(1), -1).permute(0, 2, 1)
                token = self.sam_adapter_2(self.gelu(self.sam_adapter_1(token).permute(0, 2, 1))).permute(0, 2, 1)
            elif name == 'EVA':
                token = token.view(token.size(0), token.size(1), -1).permute(0, 2, 1)
                token = self.eva_adapter_2(self.gelu(self.eva_adapter_1(token).permute(0, 2, 1))).permute(0, 2, 1)
            elif name == 'Pix2Struct':
                token = token.view(token.size(0), token.size(1), -1).permute(0, 2, 1)
                token = self.pix2struct_adapter_2(self.gelu(self.pix2struct_adapter_1(token).permute(0, 2, 1))).permute(0, 2, 1)
            elif name == 'ConvNeXt':
                token = self.convnext_adapter_2(self.gelu(self.convnext_adapter_1(token).permute(0, 2, 1))).permute(0, 2, 1)

            teacher_tokens[name] = token
            if cls_token is not None:
                m_v_t = cls_token[:, None, :] @ token.transpose(1, 2)
                teacher_weight[name] = m_v_t.mean(dim=(1, 2))

        return teacher_tokens, teacher_weight, token_weight

    def _compute_kd_loss(self, student_token, teacher_tokens, token_weight=None, force_teachers=None):
        loss_kd = nn.MSELoss(reduction='none')
        teacher_names = self.teachers_list if force_teachers is None else force_teachers
        losses = [loss_kd(student_token, teacher_tokens[name]) for name in teacher_names]
        losses = torch.stack(losses, dim=-1).mean(dim=2)
        if self.training_args.token_weight and token_weight is not None:
            weighted = (losses * token_weight.unsqueeze(-1)).sum(dim=1)
            losses = losses.mean(dim=1) + weighted
        else:
            losses = losses.mean(dim=1)
        return losses.mean()

    def _forward_single_branch(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        labels,
        images,
        image_sizes,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict,
    ):
        (
            branch_input_ids,
            branch_position_ids,
            branch_attention_mask,
            branch_past_key_values,
            branch_inputs_embeds,
            branch_labels,
            branch_vision_tower_output,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            image_sizes,
        )

        branch_outputs = super().forward(
            input_ids=branch_input_ids,
            attention_mask=branch_attention_mask,
            position_ids=branch_position_ids,
            past_key_values=branch_past_key_values,
            inputs_embeds=branch_inputs_embeds,
            labels=branch_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return branch_outputs, branch_vision_tower_output
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        raw_input_ids = input_ids
        raw_attention_mask = attention_mask
        raw_position_ids = position_ids
        raw_past_key_values = past_key_values
        raw_labels = labels

        if inputs_embeds is None:
            if self.config.kd['kd_mode'] and not self.generate_mode:
                images_processed = self.process_images(images,self.image_process)
                (
                    input_ids,
                    position_ids, 
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    vision_tower_output, # (last_hidden_state,pooler_output,hidden_states,attentions)
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images_processed,
                    image_sizes
                )
            else:
                if not self.generate_mode:
                    images = self.process_images(images,self.image_process) if not isinstance(images, torch.Tensor) else images
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes
                )

        if self.training_args.finetune and not self.generate_mode and self.config.kd['kd_mode'] and self.vision_teachers is not None:
            teacher_tokens, _, token_weight = self._compute_teacher_tokens(images)

            self._set_moe_forward_mode(mode="router_no_shared_grad", router_train_only=False)
            outputs_router, vision_router = self._forward_single_branch(
                raw_input_ids,
                raw_attention_mask,
                raw_position_ids,
                raw_past_key_values,
                raw_labels,
                images_processed,
                image_sizes,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )
            student_router = vision_router.hidden_states[self.model_args.mm_vision_select_layer][:, 1:, :]
            global_kd_loss = self._compute_kd_loss(student_router, teacher_tokens, token_weight)

            self._set_moe_forward_mode(mode="shared_only", router_train_only=False)
            outputs_shared, vision_shared = self._forward_single_branch(
                raw_input_ids,
                raw_attention_mask,
                raw_position_ids,
                raw_past_key_values,
                raw_labels,
                images_processed,
                image_sizes,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )
            student_shared = vision_shared.hidden_states[self.model_args.mm_vision_select_layer][:, 1:, :]
            clip_only_loss = self._compute_kd_loss(student_shared, teacher_tokens, token_weight, force_teachers=['CLIP'])

            self._set_moe_forward_mode(mode="router", router_train_only=False)
            total_loss = (outputs_router.loss + global_kd_loss) + (outputs_shared.loss + clip_only_loss)

            return CausalLMOutputWithPast(
                loss=total_loss,
                logits=outputs_router.logits,
                past_key_values=outputs_router.past_key_values,
                hidden_states=outputs_router.hidden_states,
                attentions=outputs_router.attentions,
            )
        
        if self.generate_mode or not self.config.kd['kd_mode']:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        loss = outputs.loss
        batch_size = inputs_embeds.shape[0]
        if self.config.moe['moe_encoder'] and self.config.moe['moe_encoder_num_experts'] > 1 and self.training_args.tune_encoder:
            clip_mlp_routing_probs = torch.stack([r[0] for r in vision_tower_output.routings], dim=0) # [layer, batch, seq_len, num_experts]
            clip_mlp_routing_idxes = torch.stack([r[1] for r in vision_tower_output.routings], dim=0).detach()

            clip_mlp_expert_balancing_loss = 0.
            batch_size = clip_mlp_routing_probs.shape[1]
            for i in range(batch_size):
                probs_i = clip_mlp_routing_probs[:,i, :].reshape(-1, self.config.moe['moe_encoder_num_experts'])
                idxes_i = clip_mlp_routing_idxes[:,i, :].reshape(-1, self.config.moe['moe_encoder_num_experts'])

                clip_mlp_expert_balancing_loss += (probs_i.mean(0) * idxes_i.mean(0)).sum()
            
            balancing_loss = clip_mlp_expert_balancing_loss / batch_size
            loss = loss + balancing_loss * self.training_args.moe_encoder_balance_w
    
        torch.cuda.empty_cache()
        kd_loss = 0.0
        if self.vision_teachers is not None and self.config.kd['kd_mode']:
            teacher_tokens, _, token_weight = self._compute_teacher_tokens(images)
            student_token = vision_tower_output.hidden_states[self.model_args.mm_vision_select_layer][:, 1:, :]

            if self.training_args.finetune:
                kd_loss = self._compute_kd_loss(student_token, teacher_tokens, token_weight)
                clip_only_loss = self._compute_kd_loss(student_token, teacher_tokens, token_weight, force_teachers=['CLIP'])
                loss = loss + kd_loss + clip_only_loss
            else:
                routed_losses = 0.0
                routed_names = [name for name in self.teachers_list if name != 'CLIP']
                for expert_idx, t_name in enumerate(routed_names):
                    self._set_moe_forward_mode(mode="forced_expert", forced_expert_idx=expert_idx)
                    routed_losses = routed_losses + self._compute_kd_loss(student_token, teacher_tokens, token_weight, force_teachers=[t_name])

                self._set_moe_forward_mode(mode="shared_only")
                shared_kd = self._compute_kd_loss(student_token, teacher_tokens, token_weight, force_teachers=routed_names)

                self._set_moe_forward_mode(mode="router", router_train_only=True)
                router_kd = self._compute_kd_loss(student_token, teacher_tokens, token_weight)
                self._set_moe_forward_mode(mode="router", router_train_only=False)
                kd_loss = (routed_losses + shared_kd + router_kd) * self.training_args.kd_w
                loss = loss + kd_loss
       
        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            images = self.process_images(images) if not isinstance(images, torch.Tensor) else images
            if self.config.kd['kd_mode']:
                (
                    inputs,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _,
                    vision_tower_output, # (last_hidden_state,pooler_output,hidden_states,attentions)
                ) = self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes
                )
            else:
                (
                    inputs,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _
                ) = self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes
                )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


class EvalMoVELlavaLlamaForCausalLM(MoVELlavaLlamaForCausalLM):
    config_class = MoVELlavaLlamaConfig

    def __init__(self, config):
        super(EvalMoVELlavaLlamaForCausalLM, self).__init__(config)
        self.generate_mode = True

        # self.model.get_vision_tower() = self.model.get_vision_tower()
        if not self.model.get_vision_tower().is_loaded:
            self.model.get_vision_tower().load_model()
        self.image_process = self.model.get_vision_tower().image_processor

        if self.config.moe['moe_encoder']:
            # replace origin_model with moe_model
            origin_model = self.model.get_vision_tower().vision_tower.vision_model
            moe_model = MoECLIPVisionTransformer(origin_model.config)
            self.model.get_vision_tower().vision_tower.vision_model = moe_model

            # replace origin_encoder with moe_encoder
            origin_encoder = self.model.get_vision_tower().vision_tower.vision_model.encoder
            moe_encoder = MoECLIPEncoder(origin_encoder.config)
            self.model.get_vision_tower().vision_tower.vision_model.encoder = moe_encoder

            # replace origin_layer with moe_layer
            for i, layer in enumerate(self.model.get_vision_tower().vision_tower.vision_model.encoder.layers):
                moe_layer = MoECLIPEncoderLayer(self.model.get_vision_tower().vision_tower.vision_model.encoder.config)
                self.model.get_vision_tower().vision_tower.vision_model.encoder.layers[i] = moe_layer

            # replace origin_mlp with moe_mlp
            for i, layer in enumerate(self.model.get_vision_tower().vision_tower.vision_model.encoder.layers):
                origin_mlp = layer.mlp
                self.model.get_vision_tower().vision_tower.vision_model.encoder.layers[i].mlp = \
                    MoE_LoRA_CLIP(
                        args=self.config,
                        lora_rank=self.config.moe['moe_encoder_lora_rank'],
                        lora_alpha=self.config.moe['moe_encoder_lora_alpha'],
                        num_experts=self.config.moe['moe_encoder_num_experts'],
                        dense_moe=self.config.moe['moe_encoder_dense'],
                        original_module=origin_mlp
                    )
        self.model.get_vision_tower().requires_grad_(False)

AutoConfig.register("move_llava_llama", MoVELlavaLlamaConfig)
AutoModelForCausalLM.register(MoVELlavaLlamaConfig, MoVELlavaLlamaForCausalLM)

AutoModelForCausalLM.register(MoVELlavaLlamaConfig, EvalMoVELlavaLlamaForCausalLM)
