import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import quivr as qv

try:
    import datafusion
    import polars
except ImportError:
    pytest.skip("polars and datafusion not installed", allow_module_level=True)


def new_table(N=1_000_00):
    class MyTable(qv.Table):
        x = qv.Float64Column()
        y = qv.StringColumn()
        z = qv.Int64Column()

    seed = 1234
    np.random.seed(seed)
    xs = np.random.rand(N)
    ys = np.random.choice(["a", "b", "c"], N)
    zs = np.random.randint(0, 100, N)

    data = MyTable.from_data(x=xs, y=ys, z=zs)
    return data


def sumif_pandas(data):
    df = data.to_dataframe()
    return df[df["y"] == "a"]["x"].sum()


def sumif_arrow(data):
    mask = pc.equal(data.y, "a")
    return pc.sum(pc.filter(data.x, mask))


def sumif_polars(data):
    df = polars.from_arrow(data.table)
    return df.filter(polars.col("y") == "a")["x"].sum()


def sumif_numpy(data):
    mask = np.where(data.y.to_numpy(False) == "a")
    return data.x.to_numpy(False)[mask].sum()


def sumif_datafusion(data):
    ctx = datafusion.SessionContext()
    ctx.from_arrow_table(data.table, "data")
    df = ctx.sql("select sum(x) as result from data where y = 'a'")
    return df.to_pylist()[0]["result"]


def test_equal():
    data = new_table()
    want = sumif_pandas(data)
    got = sumif_arrow(data).as_py()
    assert want == got

    got = sumif_polars(data)
    assert np.isclose(want, got)

    got = sumif_datafusion(data)
    assert np.isclose(want, got)

    got = sumif_numpy(data)
    assert np.isclose(want, got)


@pytest.mark.benchmark(group="query")
def test_sum_if_using_pandas(benchmark):
    data = new_table()
    benchmark(sumif_pandas, data)


@pytest.mark.benchmark(group="query")
def test_sum_if_using_arrow(benchmark):
    data = new_table()
    benchmark(sumif_arrow, data)


@pytest.mark.benchmark(group="query")
def test_sum_if_using_polars(benchmark):
    data = new_table()
    benchmark(sumif_polars, data)


@pytest.mark.benchmark(group="query")
def test_sum_if_using_datafusion(benchmark):
    data = new_table()
    benchmark(sumif_datafusion, data)


@pytest.mark.benchmark(group="query")
def test_sum_if_using_numpy(benchmark):
    data = new_table()
    benchmark(sumif_numpy, data)


class InnerTable(qv.Table):
    x = qv.Float64Column()
    y = qv.StringColumn()
    z = qv.Int64Column()


class OuterTable(qv.Table):
    x = qv.Float64Column()
    y = qv.StringColumn()
    z = qv.Int64Column()
    inner = InnerTable.as_column()


def new_outer_table(N=1_000_000):
    seed = 1234
    np.random.seed(seed)
    xs = np.random.rand(N)
    ys = np.random.choice(["a", "b", "c"], N)
    zs = np.random.randint(0, 100, N)

    inner = InnerTable.from_data(x=xs, y=ys, z=zs)
    data = OuterTable.from_data(x=xs, y=ys, z=zs, inner=inner)
    return data


def sumif_nested_pandas(data):
    df = data.to_dataframe()
    return df[df["inner.y"] == "a"]["x"].sum()


def sumif_nested_arrow(data):
    mask = pc.equal(data.inner.y, "a")
    return pc.sum(pc.filter(data.x, mask))


def sumif_nested_polars(data):
    df = polars.from_arrow(data.table)
    return df.filter(polars.col("inner").struct.field("y") == "a")["x"].sum()


def sumif_nested_numpy(data):
    mask = np.where(data.inner.y.to_numpy(False) == "a")
    return data.x.to_numpy(False)[mask].sum()


def sumif_nested_datafusion(data):
    ctx = datafusion.SessionContext()
    ctx.from_arrow_table(data.table, "data")
    df = ctx.sql("select sum(x) as result from data where inner.y = 'a'")
    return df.to_pylist()[0]["result"]


def test_sumif_nested():
    data = new_outer_table()
    want = sumif_nested_arrow(data).as_py()

    got = sumif_nested_pandas(data)
    assert want == got

    got = sumif_nested_polars(data)
    assert np.isclose(want, got)

    got = sumif_nested_datafusion(data)
    assert np.isclose(want, got)

    got = sumif_nested_numpy(data)
    assert np.isclose(want, got)


@pytest.mark.benchmark(group="query_nested")
def test_sum_if_nested_using_pandas(benchmark):
    data = new_outer_table()
    benchmark(sumif_nested_pandas, data)


@pytest.mark.benchmark(group="query_nested")
def test_sum_if_nested_using_arrow(benchmark):
    data = new_outer_table()
    benchmark(sumif_nested_arrow, data)


@pytest.mark.benchmark(group="query_nested")
def test_sum_if_nested_using_polars(benchmark):
    data = new_outer_table()
    benchmark(sumif_nested_polars, data)


@pytest.mark.benchmark(group="query_nested")
def test_sum_if_nested_using_datafusion(benchmark):
    data = new_outer_table()
    benchmark(sumif_nested_datafusion, data)


@pytest.mark.benchmark(group="query_nested")
def test_sum_if_nested_using_numpy(benchmark):
    data = new_outer_table()
    benchmark(sumif_nested_numpy, data)


class TableWithListColumn(qv.Table):
    x = qv.Float64Column()
    y = qv.StringColumn()
    z = qv.Int64Column()
    covariance = qv.Column(pa.fixed_shape_tensor(pa.float64(), (3, 3)))

    def cov_matrix(self):
        return self.covariance.combine_chunks().to_numpy_ndarray()

    def sigmas(self):
        return np.sqrt(np.diagonal(self.cov_matrix(), axis1=1, axis2=2))


def new_table_with_list_column(N=1_000_000):
    seed = 1234
    np.random.seed(seed)
    xs = np.random.rand(N)
    ys = np.random.choice(["a", "b", "c"], N)
    zs = np.random.randint(0, 100, N)
    covariances = pa.FixedShapeTensorArray.from_numpy_ndarray(np.random.rand(N, 3, 3))

    data = TableWithListColumn.from_data(x=xs, y=ys, z=zs, covariance=covariances)
    return data


def sigmasum_quivr(data):
    return data.sigmas().sum()


def sigmasum_arrow(data):
    xx = pc.list_element(data.covariance.combine_chunks().storage, 0)
    yy = pc.list_element(data.covariance.combine_chunks().storage, 4)
    zz = pc.list_element(data.covariance.combine_chunks().storage, 8)
    return pc.sum(pc.add(pc.sqrt(xx), pc.add(pc.sqrt(yy), pc.sqrt(zz))))


def test_sigmasum():
    data = new_table_with_list_column()
    want = sigmasum_quivr(data)

    got = data.sigmas().sum()
    assert np.isclose(want, got)

    got = sigmasum_arrow(data).as_py()
    assert np.isclose(want, got)


@pytest.mark.benchmark(group="sigmasum")
def test_sigmasum_using_quivr(benchmark):
    data = new_table_with_list_column()
    benchmark(sigmasum_quivr, data)


@pytest.mark.benchmark(group="sigmasum")
def test_sigmasum_using_arrow(benchmark):
    data = new_table_with_list_column()
    benchmark(sigmasum_arrow, data)
