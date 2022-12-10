def coefficient_plot(model, feature_labels=None, top=None, sort=True, xlab="Feature", ylab="Coefficient", title="Model Coefficients"):
    if feature_labels is None:
        feature_labels = ["Feature {}".format(i) for i in range(len(model.coef_[0]))]
    coefficients = model.coef_[0]
    if top != None:
        top_idx = np.argsort(np.abs(coefficients))[::-1][:top]
        coefficients = coefficients[top_idx]
        feature_labels = feature_labels[top_idx]
    with plt.rc_context(rc):
        coefs = pd.DataFrame(coefficients, columns=["Coefficients"], index=feature_labels)
        if sort:
            coefs = coefs.sort_values("Coefficients", ascending=False)
        coefs.plot(kind="bar", figsize=(16, 8), legend=None)
        plt.title(title, fontsize=18)
        plt.axhline(y=0, color=".5")
        plt.subplots_adjust(left=0.3)
        plt.xlabel(xlab, fontsize=16)
        plt.ylabel(ylab, fontsize=16)