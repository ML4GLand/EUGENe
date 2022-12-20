def _create_kernel_matrix(
    size: tuple, motifs: Dict[str, Motif], convert_to_pwm=True
) -> np.ndarray:
    if len(size) != 3:
        raise RuntimeError("Kernel matrix size must be a tuple of length 3")
    kernel = torch.zeros(size)
    torch.nn.init.xavier_uniform_(kernel)
    for i, motif_id in enumerate(motifs):
        motif = motifs[motif_id]
        if convert_to_pwm:
            new_weight = torch.tensor(
                motif.pfm[: min(len(motif), kernel.shape[2]), :] / 0.25
            ).transpose(0, 1)
        else:
            new_weight = torch.tensor(
                motif.pfm[: min(len(motif), kernel.shape[2]), :]
            ).transpose(0, 1)
        kernel[i, :, : min(len(motif), kernel.shape[2])] = new_weight
    return kernel





def info_content(pwm, transpose=False, bg_gc=0.415):
    ''' Compute PWM information content.
    In the original analysis, I used a bg_gc=0.5. For any
    future analysis, I ought to switch to the true hg19
    value of 0.415.
    '''
    pseudoc = 1e-9

    if transpose:
        pwm = np.transpose(pwm)

    bg_pwm = [1-bg_gc, bg_gc, bg_gc, 1-bg_gc]

    ic = 0
    for i in range(pwm.shape[0]):
        for j in range(4):
            # ic += 0.5 + pwm[i][j]*np.log2(pseudoc+pwm[i][j])
            ic += -bg_pwm[j]*np.log2(bg_pwm[j]) + pwm[i][j]*np.log2(pseudoc+pwm[i][j])

    return ic