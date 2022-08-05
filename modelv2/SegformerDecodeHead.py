from transformers import SegformerPreTrainedModel, SegformerConfig
from torch import nn, cat, Tensor
import math

class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, config: SegformerConfig, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states

class SegformerDecodeHead(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)
        #self.linear_c = nn.ModuleList(mlps.reverse())

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )

        self.linear_fuse_2_hidden_states = nn.Conv2d(
            in_channels=config.decoder_hidden_size * 2,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )

        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        self.config = config

    def forward(self, encoder_hidden_states):
        batch_size = encoder_hidden_states[-1].shape[0]
        
        all_hidden_states = ()
        #print(encoder_hidden_states[0].shape)
        #print(encoder_hidden_states[1].shape)
        #print(encoder_hidden_states[2].shape)
        #print(encoder_hidden_states[3].shape)
        #print(encoder_hidden_states[4].shape)
        #input()
        # MY VERSION
              
        #reversed(encoder_hidden_states)
        #FROM 
        #0. torch.Size([8, 32, 128, 128])
        #1. torch.Size([8, 64, 64, 64])
        #2. torch.Size([8, 160, 32, 32])
        #3. torch.Size([8, 256, 16, 16])
    
        #TO
        #0. torch.Size([8, 256, 16, 16])
        #1. torch.Size([8, 160, 32, 32])
        #2. torch.Size([8, 64, 64, 64])
        #3. torch.Size([8, 32, 128, 128])
        
        '''
        for idx, (encoder_hidden_state, mlp) in reversed(list(enumerate(zip(encoder_hidden_states, self.linear_c)))):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            if idx==3:
                # 1. First, multi-level features Fi from the MiT encoder goes through an MLP layer to unify the channel dimension
                height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
                encoder_hidden_state = mlp(encoder_hidden_state)
                #print(encoder_hidden_state.shape)
                encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
                encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
                # Partendo dall'ultimo... es. H/32xW/32
                # 2. Features are upsampled to the previous encoder block size
                encoder_hidden_state = nn.functional.interpolate(
                    encoder_hidden_state, size=encoder_hidden_states[idx-1].size()[2:], mode="bilinear", align_corners=False
                )
                

                all_hidden_states += (encoder_hidden_state,)
            else:
                # 1. First, multi-level features Fi from the MiT encoder goes through an MLP layer to unify the channel dimension
                height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
                encoder_hidden_state = mlp(encoder_hidden_state)
                
                encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
                encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
                
                all_hidden_states += (encoder_hidden_state,)
                #print(all_hidden_states[0].shape)
                #print(all_hidden_states[1].shape)
                #fuse the concatenated features
                hidden_states = self.linear_fuse_2_hidden_states(cat(all_hidden_states[::-1], dim=1))
                hidden_states = self.batch_norm(hidden_states)
                hidden_states = self.activation(hidden_states)
                fused_hidden_states = self.dropout(hidden_states)

                #print("fused: ", fused_hidden_states.shape)

                if idx!=0:
                    #print(idx)
                    # 2. Features are upsampled to the previous encoder block size
                    upsampled_hidden_states = nn.functional.interpolate(
                        fused_hidden_states, size=encoder_hidden_states[idx-1].size()[2:], mode="bilinear", align_corners=False
                    )
                    #print("upsampled: ", upsampled_hidden_states.shape)
                    all_hidden_states = ()
                    all_hidden_states += (upsampled_hidden_states,)

        logits = self.classifier(fused_hidden_states)
        '''
        ###########################
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )


            # 1. First, multi-level features Fi from the MiT encoder go through an MLP layer to unify the channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # 2. Features are upsampled to 1/4th and concatenated togheter
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            # concatenate
            all_hidden_states += (encoder_hidden_state,)
        
        # ALL_HIDDEN_STATES SIZES ARE:
        #torch.Size([8, 256, 128, 128])
        #torch.Size([8, 256, 128, 128])
        #torch.Size([8, 256, 128, 128])
        #torch.Size([8, 256, 128, 128])

        # 3. fuse the concatenated features
        hidden_states = self.linear_fuse(cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # 4. MLP layer taking the fused features to make the predictions
        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)
        
        return logits