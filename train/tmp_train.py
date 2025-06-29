        # Stage 2: Full structure reconstruction from frozen encoder
        B, L, H, _ = batch.masked_data['coords'].shape
        
        # Create coord_mask for GA/RA models
        unmasked_elements = ~batch.masks['coords'] & ~batch.beospank['coords']
        assert unmasked_elements.any(dim=1).all()
        
        model.train(True)
        model.encoder.eval()  # Encoder is frozen
        
        with torch.set_grad_enabled(True):
            # Forward pass through frozen encoder to get z_q
            with torch.no_grad():
                four_atom = batch.masked_data['coords'][:, :, :4, :]  # [B, L, 4, 3] for encoder
                three_atom = batch.masked_data['coords'][:, :, :3, :]  # [B, L, 3, 3]
                if model.cfg.model_type in ("GA", "RA"): 
                    z_q, _ = model.encoder(three_atom, four_atom, unmasked_elements)
                else: 
                    z_q, _ = model.encoder(three_atom)
            
            # Zero out BOS/EOS/PAD positions in z_q
            z_q[batch.beospank['coords']] = 0.0
            
            # Concatenate z_q with seq_tokens along last dimension
            # z_q: [B, L, fsq_dim], seq_tokens: [B, L] -> [B, L, 1]
            seq_tokens_float = batch.masked_data['seq'].unsqueeze(-1).float()  # [B, L, 1]
            decoder_input = torch.cat([z_q, seq_tokens_float], dim=-1)  # [B, L, fsq_dim + 1]
            
            # Decoder forward pass
            if model.cfg.model_type in ("GA", "RA"): 
                x_rec = model.decoder(decoder_input, four_atom, unmasked_elements)
            else: 
                x_rec = model.decoder(decoder_input)
            
            # x_rec is [B, L, 14, 3] for stage 2
            # Compute loss on all valid positions (no masking in stage 2)
            pts_pred = []; pts_true = []
            for batch_idx in range(B):
                real_residues = torch.arange(L, device=device)[unmasked_elements[batch_idx]]
                pred_coords = x_rec[batch_idx][real_residues]  # [M, 14, 3] 
                true_coords = batch.masked_data['coords'][batch_idx][real_residues]  # [M, 14, 3]

                # Flatten to [1, M*14, 3]
                pts_pred.append(pred_coords.reshape(1, -1, 3))
                pts_true.append(true_coords.reshape(1, -1, 3))
            
            # Compute squared Kabsch RMSD loss
            if pts_pred:
                loss = squared_kabsch_rmsd_loss(pts_pred, pts_true)
                # Also compute regular Kabsch RMSD for reporting
                with torch.no_grad():
                    rmsd = kabsch_rmsd_loss(pts_pred, pts_true)
            else:
                loss = torch.tensor(0.0, device=device)
                rmsd = torch.tensor(0.0, device=device)
            
            # Backward pass (only decoder parameters will update)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Return metrics
            return {'loss': loss.item(), 'rmsd': rmsd.item()}