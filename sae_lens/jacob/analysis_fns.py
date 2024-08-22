import torch
from datasets import load_dataset
from transformer_lens import utils
import gc
from tqdm import tqdm


# GET DATA
def remove_entries_with_bos(model, tokens):
    new_tokens = []
    for i in range(tokens.shape[0]):
        if not model.tokenizer.bos_token_id in tokens[i, 1:]:
            new_tokens.append(tokens[i, :])
    new_tokens = torch.stack(new_tokens)
    return new_tokens

def get_owt_and_spec_tokens(model, spec_path, remove_bos=True, ctx_length=128):

    # Get owt tokens
    data = load_dataset("stas/openwebtext-10k", split="train[:20%]")
    tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=ctx_length)
    tokenized_data = tokenized_data.shuffle(42)
    owt_tokens = tokenized_data["tokens"][:200_000]
    if remove_bos:
        owt_tokens = remove_entries_with_bos(model, owt_tokens)

    print("owt_tokens has shape", owt_tokens.shape)
    print("total number of tokens:", int(owt_tokens.numel()//1e6), "million")
    print()

    # Get speciaized tokens
    data = load_dataset(spec_path, split="train")
    # Use filter function to remove null entries
    def remove_null_entries(example):
        return all(value is not None and value != '' for value in example.values())
    data = data.filter(remove_null_entries)
    tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=ctx_length)
    tokenized_data = tokenized_data.shuffle(42)
    spec_tokens = tokenized_data["tokens"][:100_000]
    if remove_bos:
        spec_tokens = remove_entries_with_bos(model, spec_tokens)
    print("spec_tokens has shape", spec_tokens.shape)
    print("total number of tokens:", int(spec_tokens.numel()//1e6), "million")

    # clean up
    del tokenized_data, data
    gc.collect()

    return owt_tokens, spec_tokens


# Sweeping functions
def to_str(i):
    if i ==0: 
        return "gsae"
    elif i == 1:
        return "ssae"
    else:
        print("sae_list too large")


def get_l0_freqs_loss_fvu(
        model,
        sae_list,  # "clean" or [gsae] or [gsae, ssae]
        tokens, 
        num_tokens = 100_000, 
        batch_size=128,
        start_pos=2
        ):
    '''Computes L0, feature frequencies, the patched CE losses, and the Fraction of Variance Unexplained (FVU) for a given list of SAEs.
    The outputs of SAEs in the list are summed when patching.'''

    # make sure all SAEs have same hook point
    if sae_list != "clean":
        hook_pt_list = [sae.cfg.hook_name for sae in sae_list]
        assert len(set(hook_pt_list)) == 1, "All SAEs must have the same hook_name"
        hook_pt = hook_pt_list[0]
    else:
        # if not patching any SAEs, we still set a dummy hook - it won't be used
        hook_pt = 'blocks.0.hook_resid_pre'

    # Initialize running variables
    total_fvu = {'all_saes' : 0}
    if sae_list != "clean":
        total_freqs = {i : torch.zeros(sae.cfg.d_sae).cuda() 
                for i, sae in enumerate(sae_list)}
    else:
        total_freqs = {}
    all_losses = []
    
    # define hook fn to patch in SAE reconstructions, as well as cache the FVU and L0
    def patch_hook(act, hook):              
        # option to run the model clean
        if sae_list == "clean":
            return act
        # patch in SAE reconstructions
        else:
            out = torch.zeros_like(act)
            out[:, :start_pos] = act[:, :start_pos]
            for i, sae in enumerate(sae_list):
                feature_acts = sae.encode(act)
                total_freqs[i] += (feature_acts > 0).float()[:, start_pos:].mean([0,1])
                out[:, start_pos:] += sae.decode(feature_acts)[:, start_pos:]

            total_fvu['all_saes'] += ((out - act).pow(2)[:,start_pos:].sum(-1) / (act - act.mean(0)).pow(2)[:,start_pos:].sum(-1)).mean().item()
            return out

    num_batches = num_tokens // (tokens.shape[1] * batch_size)

    for b in tqdm(range(num_batches)):
        # get batch
        batch = tokens[b*batch_size:(b+1)*batch_size].cuda()
        losses = model.run_with_hooks(
            batch,
            return_type="loss",
            loss_per_token=True,
            fwd_hooks = [(hook_pt, patch_hook)]
        )[:,start_pos:]

        all_losses.append(losses)
        
    all_losses = torch.cat(all_losses)
    freqs = {to_str(i) : count / num_batches for i, count in total_freqs.items()}
    l0 = sum([freqs.sum() for _, freqs in freqs.items()])
    return  l0, freqs, all_losses, total_fvu['all_saes'] / num_batches



def sweep(
        model,
        list_of_sae_lists, 
        tokens, 
        ceiling_losses, 
        clean_losses, 
        ceiling_fvu,
        num_tokens = 100_000):

    l0_list = []
    freqs_list = []
    score_list = []
    fvu_recovered_list = []
    
    for sae_list in list_of_sae_lists:

        l0, freqs, all_losses, fvu = get_l0_freqs_loss_fvu(model, sae_list, tokens, num_tokens=num_tokens)
        l0_list.append(l0.item())
        # compute loss recovered score
        mask = ceiling_losses - clean_losses > 0.001 # ignore cases where there was no loss to recover in the first place
        
        all_scores = (ceiling_losses[mask].mean() - all_losses[mask].mean()) / (ceiling_losses[mask].mean() - clean_losses[mask].mean())
        
        # print(f"\nceiling_losses {ceiling_losses.mean().item():.3f}")
        # print(f"patched_losses {all_losses.mean().item():.3f}")
        # print(f"clean_losses {clean_losses.mean().item():.3f}\n")

        score_list.append(all_scores.mean().item())

        # compute variance explained
        fvu_recovered = (ceiling_fvu - fvu) / ceiling_fvu
        fvu_recovered_list.append(fvu_recovered)

        freqs_list.append(freqs)
        
    return l0_list, freqs_list, score_list, fvu_recovered_list




# PLOTTING
import matplotlib.pyplot as plt
import os
def get_freq_plots(ssae_owt_freqs, direct_owt_freqs, gsae_ft_owt_freqs,
                   ssae_spec_freqs, direct_spec_freqs, gsae_ft_spec_freqs, 
                   subject):   
    # DECODERS
    # choose medium-sparse SSAE and compare frequencies to GSAE
    gsae_logfreqs = (ssae_owt_freqs[1]['gsae'] + 1e-8).log10().cpu()
    ssae_logfreqs = (ssae_owt_freqs[1]['ssae'] + 1e-8).log10().cpu()
    direct_logfreqs = (direct_owt_freqs[1]['gsae'] + 1e-8).log10().cpu()
    gsae_ft_logfreqs = (gsae_ft_owt_freqs[1]['gsae'] + 1e-8).log10().cpu()

    # Plotting the histograms
    plt.hist(gsae_logfreqs.numpy(), bins=70, alpha=0.4, label='GSAE', density=True)
    plt.hist(ssae_logfreqs.numpy(), bins=70, alpha=0.4, label='SSAE', density=True)
    plt.hist(direct_logfreqs.numpy(), bins=70, alpha=0.4, label='Direct SAE', density=True)
    plt.hist(gsae_ft_logfreqs.numpy(), bins=70, alpha=0.4, label='GSAE Finetune', density=True)

    # Adding labels and title
    plt.xlabel('Log Frequencies')
    plt.ylabel('Probability density')
    plt.title(f'{subject[:-8]}: Histogram of GSAE and SSAE Log Frequencies on OWT')
    plt.legend()

    directory = 'plots/owt_freqs'
    filename = f'{subject}.png'
    filepath = os.path.join(directory, filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(filepath)
    plt.close()


    # choose sparsest SSAE and compare frequencies to GSAE
    gsae_logfreqs = (ssae_spec_freqs[1]['gsae'] + 1e-8).log10().cpu()
    ssae_logfreqs = (ssae_spec_freqs[1]['ssae'] + 1e-8).log10().cpu()
    direct_logfreqs = (direct_spec_freqs[1]['gsae'] + 1e-8).log10().cpu()
    gsae_ft_logfreqs = (gsae_ft_spec_freqs[1]['gsae'] + 1e-8).log10().cpu()

    # Plotting the histograms
    plt.hist(gsae_logfreqs.numpy(), bins=70, alpha=0.4, label='GSAE', density=True)
    plt.hist(ssae_logfreqs.numpy(), bins=70, alpha=0.4, label='SSAE', density=True)
    plt.hist(direct_logfreqs.numpy(), bins=70, alpha=0.4, label='Direct', density=True)
    plt.hist(gsae_ft_logfreqs.numpy(), bins=70, alpha=0.4, label='GSAE Finetune', density=True)

    # Adding labels and title
    plt.xlabel('Log Frequencies')
    plt.ylabel('Probability density')
    plt.title(f'{subject[:-8]}: Histogram of GSAE and SSAE Log Frequencies on specialized dataset')
    plt.legend()

    directory = 'plots/spec_freqs'
    filename = f'{subject}.png'
    filepath = os.path.join(directory, filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(filepath)
    plt.close()





# def get_cossim_plots(gsae, gsae_ft_list, ssae_list, 
#                      ssae_l1_list, gsae_ft_l1_list,
#                      direct_sae_list, direct_sae_l1_list,
#                      subject):
#     # DECODERS
#     # Create a single figure with three rows
#     fig, axes = plt.subplots(3, len(ssae_list), figsize=(15, 10))

#     # GSAE-ft ZOOMED (First row)
#     for i, sae in enumerate(gsae_ft_list):
#         gsae_W_dec = gsae.W_dec / gsae.W_dec.norm(dim=1, keepdim=True)
#         sae_W_dec = sae.W_dec / sae.W_dec.norm(dim=1, keepdim=True)
#         maxsims = (gsae_W_dec @ sae_W_dec.T).max(0).values.to(torch.float32).cpu().detach()
        
#         # Select the current axis in the first row
#         ax = axes[0, i]
        
#         # Plot the histogram on the current axis
#         ax.hist(maxsims, bins=100, alpha=1.0)
#         ax.set_xlabel('Max cossim')
#         ax.set_ylabel('Frequency')
#         ax.set_title(f'GSAE-ft: l1_coeff = {gsae_ft_l1_list[i]}')
#         ax.set_xlim([0, 1])
#         ax.set_ylim([0, 500])

#     # SSAE (Second row)
#     for i, sae in enumerate(ssae_list):
#         gsae_W_dec = gsae.W_dec / gsae.W_dec.norm(dim=1, keepdim=True)
#         sae_W_dec = sae.W_dec / sae.W_dec.norm(dim=1, keepdim=True)
#         maxsims = (gsae_W_dec @ sae_W_dec.T).max(0).values.to(torch.float32).cpu().detach()
        
#         # Select the current axis in the second row
#         ax = axes[1, i]
        
#         # Plot the histogram on the current axis
#         ax.hist(maxsims, bins=50, alpha=1.0)
#         ax.set_xlabel('Max cossim')
#         ax.set_ylabel('Frequency')
#         ax.set_title(f'SSAE: l1_coeff = {ssae_l1_list[i]}')
#         ax.set_xlim([0, 1])
#         ax.set_ylim([0, 500])

#     # Direct SAE (Third row)
#     for i, sae in enumerate(direct_sae_list):
#         gsae_W_dec = gsae.W_dec / gsae.W_dec.norm(dim=1, keepdim=True)
#         sae_W_dec = sae.W_dec / sae.W_dec.norm(dim=1, keepdim=True)
#         maxsims = (gsae_W_dec @ sae_W_dec.T).max(0).values.to(torch.float32).cpu().detach()
        
#         # Select the current axis in the third row
#         ax = axes[2, i]
        
#         # Plot the histogram on the current axis
#         ax.hist(maxsims, bins=50, alpha=1.0)
#         ax.set_xlabel('Max cossim')
#         ax.set_ylabel('Frequency')
#         ax.set_title(f'Direct SAE: l1_coeff = {direct_sae_l1_list[i]}')
#         ax.set_xlim([0, 1])
#         ax.set_ylim([0, 500])

#     # Add row titles
#     # fig.text(0.5, 0.98, f'{subject[:-8]}: Max decoder cossim between GSAE-finetune & GSAE', ha='center', va='center', fontsize=16)
#     # fig.text(0.5, 0.51, f'{subject[:-8]}: Max decoder cossim between SSAE & GSAE', ha='center', va='center', fontsize=16)
#     # fig.text(0.5, 0.05, f'{subject[:-8]}: Max decoder cossim between Direct SAE & GSAE', ha='center', va='center', fontsize=16)
#     # Add row titles with some space between the rows of the subplot
#     fig.text(0.5, 0.98, f'{subject[:-8]}: Max decoder cossim between GSAE-finetune & GSAE', ha='center', va='center', fontsize=16)
#     fig.text(0.5, 0.66, f'{subject[:-8]}: Max decoder cossim between SSAE & GSAE', ha='center', va='center', fontsize=16)
#     fig.text(0.5, 0.34, f'{subject[:-8]}: Max decoder cossim between Direct SAE & GSAE', ha='center', va='center', fontsize=16)

#     # Adjust the layout
#     plt.tight_layout()

#     # Add more space between rows
#     plt.subplots_adjust(hspace=0.4)

#     directory = 'plots/dec_cossimss'
#     filename = f'{subject}.png'
#     filepath = os.path.join(directory, filename)
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     plt.savefig(filepath)
#     plt.close()


#     # ENCODERS
#     # Create a single figure with three rows
#     fig, axes = plt.subplots(3, len(ssae_list), figsize=(15, 10))

#     # GSAE-ft ZOOMED (First row)
#     for i, sae in enumerate(gsae_ft_list):
#         gsae_W_enc = gsae.W_enc / gsae.W_enc.norm(dim=0, keepdim=True)
#         sae_W_enc = sae.W_enc / sae.W_enc.norm(dim=0, keepdim=True)
#         maxsims = (gsae_W_enc.T @ sae_W_enc).max(0).values.to(torch.float32).cpu().detach()
        
#         # Select the current axis in the first row
#         ax = axes[0, i]
        
#         # Plot the histogram on the current axis
#         ax.hist(maxsims, bins=100, alpha=1.0)
#         ax.set_xlabel('Max cossim')
#         ax.set_ylabel('Frequency')
#         ax.set_title(f'GSAE-ft: l1_coeff = {gsae_ft_l1_list[i]}')
#         ax.set_xlim([0, 1])
#         ax.set_ylim([0, 500])

#     # SSAE (Second row)
#     for i, sae in enumerate(ssae_list):
#         gsae_W_enc = gsae.W_enc / gsae.W_enc.norm(dim=0, keepdim=True)
#         sae_W_enc = sae.W_enc / sae.W_enc.norm(dim=0, keepdim=True)
#         maxsims = (gsae_W_enc.T @ sae_W_enc).max(0).values.to(torch.float32).cpu().detach()
        
#         # Select the current axis in the second row
#         ax = axes[1, i]
        
#         # Plot the histogram on the current axis
#         ax.hist(maxsims, bins=50, alpha=1.0)
#         ax.set_xlabel('Max cossim')
#         ax.set_ylabel('Frequency')
#         ax.set_title(f'SSAE: l1_coeff = {ssae_l1_list[i]}')
#         ax.set_xlim([0, 1])
#         ax.set_ylim([0, 500])

#     # Direct SAE (Third row)
#     for i, sae in enumerate(direct_sae_list):
#         gsae_W_enc = gsae.W_enc / gsae.W_enc.norm(dim=0, keepdim=True)
#         sae_W_enc = sae.W_enc / sae.W_enc.norm(dim=0, keepdim=True)
#         maxsims = (gsae_W_enc.T @ sae_W_enc).max(0).values.to(torch.float32).cpu().detach()
        
#         # Select the current axis in the third row
#         ax = axes[2, i]
        
#         # Plot the histogram on the current axis
#         ax.hist(maxsims, bins=50, alpha=1.0)
#         ax.set_xlabel('Max cossim')
#         ax.set_ylabel('Frequency')
#         ax.set_title(f'Direct SAE: l1_coeff = {direct_sae_l1_list[i]}')
#         ax.set_xlim([0, 1])
#         ax.set_ylim([0, 500])


#     # Add row titles with some space between the rows of the subplot
#     fig.text(0.5, 0.98, f'{subject[:-8]}: Max encoder cossim between GSAE-finetune & GSAE', ha='center', va='center', fontsize=16)
#     fig.text(0.5, 0.66, f'{subject[:-8]}: Max encoder cossim between SSAE & GSAE', ha='center', va='center', fontsize=16)
#     fig.text(0.5, 0.34, f'{subject[:-8]}: Max encoder cossim between Direct SAE & GSAE', ha='center', va='center', fontsize=16)

#     # Adjust the layout
#     plt.tight_layout()

#     # Add more space between rows
#     plt.subplots_adjust(hspace=0.4)

#     directory = 'plots/enc_cossimss'
#     filename = f'{subject}.png'
#     filepath = os.path.join(directory, filename)
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     plt.savefig(filepath)
#     plt.show()
#     plt.close()

def get_cossim_plots(gsae, gsae_ft_list, ssae_list, 
                     ssae_l1_list, gsae_ft_l1_list,
                     direct_sae_list, direct_sae_l1_list,
                     subject):
    # Create a single figure with three rows and two columns
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    # Function to plot histogram
    def plot_histogram(ax, data, title, color):
        ax.hist(data, bins=50, alpha=0.4, color=color)
        ax.set_xlabel('Max cossim')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 500])     

    # Get the central index
    central_index = len(ssae_list) // 2

    # DECODERS (First column)
    # GSAE-ft
    gsae_W_dec = gsae.W_dec / gsae.W_dec.norm(dim=1, keepdim=True)
    sae_W_dec = gsae_ft_list[central_index].W_dec / gsae_ft_list[central_index].W_dec.norm(dim=1, keepdim=True)
    maxsims = (gsae_W_dec @ sae_W_dec.T).max(0).values.to(torch.float32).cpu().detach()
    plot_histogram(axes[0, 0], maxsims, '', 'red')

    # SSAE
    sae_W_dec = ssae_list[central_index].W_dec / ssae_list[central_index].W_dec.norm(dim=1, keepdim=True)
    maxsims = (gsae_W_dec @ sae_W_dec.T).max(0).values.to(torch.float32).cpu().detach()
    plot_histogram(axes[1, 0], maxsims, '', 'darkgoldenrod')

    # Direct SAE
    sae_W_dec = direct_sae_list[central_index].W_dec / direct_sae_list[central_index].W_dec.norm(dim=1, keepdim=True)
    maxsims = (gsae_W_dec @ sae_W_dec.T).max(0).values.to(torch.float32).cpu().detach()
    plot_histogram(axes[2, 0], maxsims, '', 'green')

    # ENCODERS (Second column)
    # GSAE-ft
    gsae_W_enc = gsae.W_enc / gsae.W_enc.norm(dim=0, keepdim=True)
    sae_W_enc = gsae_ft_list[central_index].W_enc / gsae_ft_list[central_index].W_enc.norm(dim=0, keepdim=True)
    maxsims = (gsae_W_enc.T @ sae_W_enc).max(0).values.to(torch.float32).cpu().detach()
    plot_histogram(axes[0, 1], maxsims, '', 'red')

    # SSAE
    sae_W_enc = ssae_list[central_index].W_enc / ssae_list[central_index].W_enc.norm(dim=0, keepdim=True)
    maxsims = (gsae_W_enc.T @ sae_W_enc).max(0).values.to(torch.float32).cpu().detach()
    plot_histogram(axes[1, 1], maxsims, '', 'darkgoldenrod')

    # Direct SAE
    sae_W_enc = direct_sae_list[central_index].W_enc / direct_sae_list[central_index].W_enc.norm(dim=0, keepdim=True)
    maxsims = (gsae_W_enc.T @ sae_W_enc).max(0).values.to(torch.float32).cpu().detach()
    plot_histogram(axes[2, 1], maxsims, '', 'green')

    # Set column titles
    axes[0, 0].text(0.5, 1.1, 'Decoder', ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=14)
    axes[0, 1].text(0.5, 1.1, 'Encoder', ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=14)

    # Set row titles
    for i, title in enumerate(['GSAE-finetune', 'SSAE', 'Direct SAE']):
        fig.text(0.08, 0.77 - i*0.31, title, ha='right', va='center', rotation='vertical', fontsize=14)

    # Set overall title
    fig.suptitle(f'{subject[:-8]}: Max cossim between new feature and old GSAE features', fontsize=16)

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)

    # Save the figure
    directory = 'plots/cossims'
    filename = f'{subject}_combined.png'
    filepath = os.path.join(directory, filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(filepath)
    plt.close()