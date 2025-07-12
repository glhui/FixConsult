import os
from typing import Optional, Union, Tuple

import torch
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput


class CodeT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, requires_grad=True):
        super().__init__(config)

        self.log_sigma1 = torch.nn.Parameter(torch.tensor(0.0, requires_grad=requires_grad))
        self.log_sigma2 = torch.nn.Parameter(torch.tensor(0.0, requires_grad=requires_grad)) # 基本方案不需要这个，ra的时候需要
        # self.log_sigma3 = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        ra_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        # kwargs.pop("num_items_in_batch", None)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=return_dict,
            **kwargs
        )

        # Compute the original T5 loss if labels are provided
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(outputs.logits.view(-1, self.config.vocab_size), labels.view(-1))
            outputs.loss = loss

            # if mask_labels is not None:
            #     mask_outputs = super().forward(
            #         input_ids=input_ids,
            #         attention_mask=attention_mask,
            #         decoder_input_ids=decoder_input_ids,
            #         decoder_attention_mask=decoder_attention_mask,
            #         labels=mask_labels,
            #         return_dict=return_dict,
            #         **kwargs
            #     )
            #     mask_loss_fct = CrossEntropyLoss(ignore_index=-100)
            #     mask_logits = mask_outputs.logits * mask_indexs.unsqueeze(-1);
            #     mask_loss = mask_loss_fct(mask_logits.view(-1, self.config.vocab_size), labels.view(-1))


        if ra_mask is not None and labels is not None:
            custom_loss_fct = CrossEntropyLoss(ignore_index=-100)
            adjusted_logits = outputs.logits * ra_mask.unsqueeze(-1)  # Ensure ra_mask is broadcastable
            custom_loss = custom_loss_fct(adjusted_logits.view(-1, self.config.vocab_size), labels.view(-1))
            weighted_loss = (
                    (loss / (2 * torch.exp(2 * self.log_sigma1))) +
                    (custom_loss / (2 * torch.exp(2 * self.log_sigma2))))
            outputs.loss = weighted_loss


        return outputs

    def save_pretrained(self, save_directory, **kwargs):
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
        # 获取 state_dict 并克隆每个参数以打破共享
        state_dict = {k: v.clone() for k, v in self.state_dict().items()}
        # 保存 CodeT5 权重
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        # 保存配置文件
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained_with_encoder(cls, save_directory, **kwargs):
        # 1. 加载配置文件
        config = T5Config.from_pretrained(save_directory)
        # 2. 实例化模型
        model = cls(config, **kwargs)
        # 3. 加载 CodeT5 部分的权重（不包含 weight_encoder）
        state_dict = torch.load(os.path.join(save_directory, "pytorch_model.bin"))
        model.load_state_dict(state_dict, strict=False)  # 允许缺少 weight_encoder 的 key
        # 检查 log_sigma1 和 log_sigma2 是否在权重文件中
        if "log_sigma1" in state_dict and "log_sigma2" in state_dict: #and "log_sigma3" in state_dict:
            print("Loading log_sigma1 and log_sigma2 log_sigma3 from checkpoint...")
        else:
            print("[Warning]: log_sigma1 and log_sigma2 not found in checkpoint!")
        return model