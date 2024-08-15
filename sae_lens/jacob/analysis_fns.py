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
    new_tokens = torch.stack(new_tokens).cuda()
    return new_tokens

def get_owt_and_spec_tokens(model, spec_path, remove_bos=True, ctx_length=128):

    # Get owt tokens
    data = load_dataset("stas/openwebtext-10k", split="train")
    tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=ctx_length)
    tokenized_data = tokenized_data.shuffle(42)
    owt_tokens = tokenized_data["tokens"][:20_000]
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
    spec_tokens = tokenized_data["tokens"][:20_000].cuda()
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
        batch = tokens[b*batch_size:(b+1)*batch_size]
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
    
    for sae_list in tqdm(list_of_sae_lists):

        l0, freqs, all_losses, fvu = get_l0_freqs_loss_fvu(model, sae_list, tokens, num_tokens=num_tokens)
        l0_list.append(l0)
        # compute loss recovered score
        mask = ceiling_losses - clean_losses > 0.001 # ignore cases where there was no loss to recover in the first place
        
        all_scores = (ceiling_losses[mask].mean() - all_losses[mask].mean()) / (ceiling_losses[mask].mean() - clean_losses[mask].mean())
        
        print(f"ceiling_losses {ceiling_losses.mean().item():.3f}")
        print(f"patched_losses {all_losses.mean().item():.3f}")
        print(f"clean_losses {clean_losses.mean().item():.3f}")

        score_list.append(all_scores.mean())

        # compute variance explained
        fvu_recovered = (ceiling_fvu - fvu) / ceiling_fvu
        fvu_recovered_list.append(fvu_recovered)

        freqs_list.append(freqs)
        
    return l0_list, freqs_list, score_list, fvu_recovered_list