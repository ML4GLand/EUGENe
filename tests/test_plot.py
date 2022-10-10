"""
Tests to make sure the plot module isn't busted
"""
import eugene as eu
import pytest
from pathlib import Path

HERE = Path(__file__).parent
eu.settings.logging_dir = f"{HERE}/_logs"


@pytest.fixture
def model():
    """
    model
    """
    model = eu.models.DeepBind(input_len=100, output_dim=1)
    return model


@pytest.fixture
def sdata(model):
    """
    sdata
    """
    sdata = eu.datasets.random1000()
    eu.pp.ohe_seqs_sdata(sdata)
    eu.pp.train_test_split_sdata(sdata)
    eu.interpret.feature_attribution_sdata(model, sdata)
    eu.interpret.generate_pfms_sdata(model, sdata)
    eu.dataload.motif.jaspar_annots_sdata(
        sdata,
        motif_names=["GATA1"]
    )
    eu.interpret.pca(
        sdata,
        uns_key="InputXGradient_imps",
    )
    return sdata


def test_histplot(sdata):
    ax = eu.pl.histplot(sdata, keys=["activity_0", "activity_1"], orient="h", hue="train_val", return_axes=True)
    ax[0].set_xlabel("ETS Activity")
    ax[1].set_xlabel("GATA Activity")


def test_boxplots(sdata):
    ax = eu.pl.boxplot(sdata, keys=['activity_0', 'activity_1', 'activity_2', 'activity_3'], groupby="train_val", return_axes=True)
    ax[0].set_xlabel("ETS Activity")
    ax[1].set_xlabel("GATA Activity")
    ax[2].set_xlabel("ETS Activity")


def test_violinplot(sdata):
    eu.pl.violinplot(sdata, keys=['activity_0', 'activity_1', 'activity_2'], groupby="train_val", hue="label_0")


def test_scatterplot(sdata):
    eu.pl.scatterplot(sdata, x=["activity_0", "activity_1"], y="activity_0")


def test_loss_curve():
    eu.pl.loss_curve()


def test_metric_curve():
    eu.pl.metric_curve(metric="r2")

def training_summary():
    eu.pl.training_summary(metric="r2")


def test_performance_scatterplot(sdata):
    eu.pl.performance_scatter(
        sdata, 
        target_keys="activity_1", 
        prediction_keys="activity_1"
    )


def test_confusion_matrix(sdata):
    eu.pl.confusion_mtx(sdata, target_key="label_0", prediction_key="activity_0", threshold=0.1, title="Test")


def test_auroc(sdata):
    eu.pl.auroc(sdata, target_keys=["label_0", "label_0"], prediction_keys=["activity_0", "activity_1"])


def test_auprc(sdata):
    eu.pl.auprc(sdata, target_keys=["label_0", "label_0"], prediction_keys=["activity_0", "activity_1"])


def test_seq_track_features(sdata):
    eu.pl.seq_track_features(sdata, seq_id=sdata.names[59], uns_key="InputXGradient_imps", pred_key="activity_0")


def test_multiseq_track_features(sdata):
    eu.pl.multiseq_track_features(sdata, seq_ids=sdata.names[:3], uns_keys="InputXGradient_imps", pred_key="activity_0")


# TODO
def test_filter_viz_seqlogo(sdata):
    pass


def test_seq_track(sdata):
    eu.pl.seq_track(sdata, seq_id=sdata.names[59], uns_key="InputXGradient_imps")


def multiseq_track(sdata):
    eu.pl.multiseq_track(sdata, seq_ids=sdata.names[:3], uns_keys="InputXGradient_imps")


def test_filter_viz(sdata):
    eu.pl.filter_viz(sdata, filter_id=0)


def multifilter_viz(sdata):
    eu.pl.multifilter_viz(sdata, filter_ids=[0, 1], num_cols=2, num_rows=1, figsize=(10, 2))


def test_pca(sdata):
    eu.pl.pca(sdata, seqsm_key="InputXGradient_imps_pca", color="r")