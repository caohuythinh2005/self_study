import torch

def tokenizer(text, encode=True, mask=None, max_seq_length=32):
    if encode:
        out = chr(2) + text + chr(3)
        out = out + "".join([chr(0) for _ in range(max_seq_length-len(out))]) # Adding padding
        out = torch.IntTensor(list(out.encode("utf-8")))
        mask = torch.ones(len(out.nonzero()))
        mask = torch.cat((mask, torch.zeros(max_seq_length-len(mask)))).type(torch.IntTensor)

    else: 
        out = [chr(x) for x in text[1:len(mask.nonzero())-1]]
        out = "".join(out)
        mask = None

    return out, mask