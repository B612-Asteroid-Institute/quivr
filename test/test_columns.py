import contextlib
import dataclasses
import datetime
import decimal
from typing import Optional

import pyarrow as pa
import pytest

import quivr as qv
import quivr.columns

current_id = 0


def incrementing_id():
    global current_id
    current_id += 1
    return str(current_id)


@contextlib.contextmanager
def global_id():
    global current_id
    current_id = 0
    yield
    current_id = 0


class MyTable(qv.Table):
    s1 = qv.StringColumn(default="defval", nullable=False)
    s2 = qv.StringColumn(default=incrementing_id, nullable=False)


def test_default_scalar_value():
    with global_id():
        t = MyTable.from_kwargs(
            s1=["a", "b", "c", None, "d"],
            s2=["a", "b", "c", None, "d"],
        )
        assert t.s1.to_pylist() == ["a", "b", "c", "defval", "d"]
        assert t.s2.to_pylist() == ["a", "b", "c", "1", "d"]


def test_default_callable_value():
    with global_id():
        t = MyTable.from_kwargs(
            s1=["a", "b", "c", None, "d"],
            s2=["a", None, None, None, "d"],
        )
        assert t.s1.to_pylist() == ["a", "b", "c", "defval", "d"]
        assert t.s2.to_pylist() == ["a", "1", "2", "3", "d"]


def test_default_when_array_missing():
    with global_id():
        t = MyTable.from_kwargs(
            s1=["a", "b", "c"],
        )
        assert t.s1.to_pylist() == ["a", "b", "c"]
        assert t.s2.to_pylist() == ["1", "2", "3"]

        t2 = MyTable.from_kwargs(
            s2=["a", "b", "c"],
        )
        assert t2.s1.to_pylist() == ["defval", "defval", "defval"]
        assert t2.s2.to_pylist() == ["a", "b", "c"]


# These tests are extremely repetitive, so there's some fancy
# programmatic generation of the test cases. The things that are
# tested are:
#
# 1. The default value can be a scalar or a callable
# 2. Scalar values that are too large, too small, or incorrectly typed
#    should fail at class definition time
# 3. Callable values that return a value that is too large, too small,
#    or incorrectly typed should fail at instance creation time
# 4. The default value should only fill in nulls, not overwrite existing
#    values (this is the use for the 'valid_value' field).


@dataclasses.dataclass
class DefaultValueTestCase:
    column_class: type
    default: object
    valid_value: Optional[object] = None
    expected_value: Optional[object] = None
    should_error: bool = False

    # Any extra kwargs to pass to the column instance creation, for
    # example to set the size parameter of a fixed-size binary type
    column_kwargs: dict = dataclasses.field(default_factory=dict)

    def id(self) -> str:
        if self.should_error:
            return f"{self.column_class.__name__}_{self.default}_should_error"
        return f"{self.column_class.__name__}_{self.default}_no_error"


def generate_int_testcases():
    """Generator for integer test cases"""
    cases = []
    for column_class in [
        qv.Int8Column,
        qv.Int16Column,
        qv.Int32Column,
        qv.Int64Column,
        qv.UInt8Column,
        qv.UInt16Column,
        qv.UInt32Column,
        qv.UInt64Column,
    ]:
        if pa.types.is_signed_integer(column_class.primitive_dtype):
            min_val = -(2 ** (column_class.primitive_dtype.bit_width - 1))
            max_val = 2 ** (column_class.primitive_dtype.bit_width - 1) - 1
        else:
            min_val = 0
            max_val = 2**column_class.primitive_dtype.bit_width - 1
        cases.append(
            DefaultValueTestCase(
                column_class=column_class,
                default=1,
                valid_value=2,
                expected_value=1,
            )
        )
        cases.append(
            DefaultValueTestCase(
                column_class=column_class,
                default=max_val + 1,
                should_error=True,
            )
        )
        cases.append(
            DefaultValueTestCase(
                column_class=column_class,
                default=min_val - 1,
                should_error=True,
            )
        )
        cases.append(
            DefaultValueTestCase(
                column_class=column_class,
                default="abc",
                should_error=True,
            )
        )
    return cases


def generate_string_testcases():
    cases = []
    for column_class in [qv.StringColumn, qv.LargeStringColumn]:
        cases.extend(
            [
                DefaultValueTestCase(
                    column_class=column_class,
                    default="abc",
                    expected_value="abc",
                    valid_value="def",
                ),
                DefaultValueTestCase(
                    column_class=column_class,
                    default=3000,
                    should_error=True,
                ),
            ]
        )

    for column_class in [qv.BinaryColumn, qv.LargeBinaryColumn]:
        cases.extend(
            [
                DefaultValueTestCase(
                    column_class=column_class,
                    default=b"abc",
                    expected_value=b"abc",
                    valid_value=b"def",
                ),
                DefaultValueTestCase(
                    column_class=column_class,
                    default=3000,
                    should_error=True,
                ),
            ]
        )
    return cases


def generate_float_testcases():
    cases = []
    for column_class in [qv.Float32Column, qv.Float64Column]:
        cases.extend(
            [
                DefaultValueTestCase(
                    column_class=column_class,
                    default=1.0,
                    valid_value=2.0,
                    expected_value=1.0,
                ),
                DefaultValueTestCase(
                    column_class=column_class,
                    default="abc",
                    should_error=True,
                ),
            ]
        )
    return cases


def generate_bool_testcases():
    return [
        DefaultValueTestCase(
            column_class=qv.BooleanColumn,
            default=True,
            valid_value=False,
            expected_value=True,
        ),
        DefaultValueTestCase(
            column_class=qv.BooleanColumn,
            default=False,
            valid_value=True,
            expected_value=False,
        ),
        DefaultValueTestCase(
            column_class=qv.BooleanColumn,
            default=1,
            should_error=True,
        ),
        DefaultValueTestCase(
            column_class=qv.BooleanColumn,
            default="abc",
            should_error=True,
        ),
    ]


def generate_fixedsize_binary_testcases():
    return [
        DefaultValueTestCase(
            column_class=qv.FixedSizeBinaryColumn,
            column_kwargs={"byte_width": 3},
            default=b"abc",
            valid_value=b"def",
            expected_value=b"abc",
        ),
        DefaultValueTestCase(
            column_class=qv.FixedSizeBinaryColumn,
            column_kwargs={"byte_width": 3},
            default=3000,
            should_error=True,
        ),
        DefaultValueTestCase(
            column_class=qv.FixedSizeBinaryColumn,
            column_kwargs={"byte_width": 3},
            default=b"ab",
            should_error=True,
        ),
    ]


def generate_decimal_testcases():
    cases = []
    for column_class in [qv.Decimal128Column, qv.Decimal256Column]:
        cases.extend(
            [
                DefaultValueTestCase(
                    column_class=column_class,
                    column_kwargs={"precision": 5, "scale": 2},
                    default=decimal.Decimal("1.23"),
                    valid_value=decimal.Decimal("2.34"),
                    expected_value=decimal.Decimal("1.23"),
                ),
                DefaultValueTestCase(
                    column_class=column_class,
                    column_kwargs={"precision": 5, "scale": 2},
                    default=3000,
                    should_error=True,
                ),
                DefaultValueTestCase(
                    column_class=column_class,
                    column_kwargs={"precision": 5, "scale": 2},
                    default=decimal.Decimal("1.234"),
                    should_error=True,
                ),
            ]
        )
    return cases


def generate_timestamp_testcases():
    return [
        DefaultValueTestCase(
            column_class=qv.TimestampColumn,
            column_kwargs={"unit": "s"},
            default=datetime.datetime(2020, 1, 1, 0, 0, 0),
            valid_value=datetime.datetime(2020, 1, 1, 0, 0, 1),
            expected_value=datetime.datetime(2020, 1, 1, 0, 0, 0),
        ),
        DefaultValueTestCase(
            column_class=qv.TimestampColumn,
            column_kwargs={"unit": "s"},
            default="hello",
            should_error=True,
        ),
        DefaultValueTestCase(
            column_class=qv.Time32Column,
            column_kwargs={"unit": "s"},
            default=datetime.time(0, 0, 0),
            valid_value=datetime.time(0, 0, 1),
            expected_value=datetime.time(0, 0, 0),
        ),
        DefaultValueTestCase(
            column_class=qv.Time32Column,
            column_kwargs={"unit": "s"},
            default="hello",
            should_error=True,
        ),
        DefaultValueTestCase(
            column_class=qv.Time64Column,
            column_kwargs={"unit": "ns"},
            default=datetime.time(0, 0, 0),
            valid_value=datetime.time(0, 0, 1),
            expected_value=datetime.time(0, 0, 0),
        ),
        DefaultValueTestCase(
            column_class=qv.Time64Column,
            column_kwargs={"unit": "ns"},
            default="hello",
            should_error=True,
        ),
    ]


@pytest.mark.parametrize(
    "test_case",
    [
        *generate_int_testcases(),
        *generate_string_testcases(),
        *generate_float_testcases(),
        *generate_bool_testcases(),
        *generate_fixedsize_binary_testcases(),
        *generate_decimal_testcases(),
        *generate_timestamp_testcases(),
    ],
    ids=lambda tc: tc.id(),
)
def test_default_values(test_case):
    if test_case.should_error:
        # Should be an error at class definition time if the default
        # value is an invalid scalar

        with pytest.raises(qv.InvalidColumnDefault):

            class MyTable(qv.Table):
                col = test_case.column_class(
                    default=test_case.default, nullable=False, **test_case.column_kwargs
                )

        # a callable which returns an invalid scalar should be an
        # error at instantiation time
        def default_value():
            return test_case.default

        class MyTable2(qv.Table):
            col = test_case.column_class(default=default_value, nullable=False, **test_case.column_kwargs)

        with pytest.raises(qv.InvalidColumnDefault):
            t = MyTable2.from_kwargs(col=[None])
        return

    # Scalar default case
    class MyTable3(qv.Table):
        col = test_case.column_class(default=test_case.default, nullable=False, **test_case.column_kwargs)

    t = MyTable3.from_kwargs(col=[None, test_case.valid_value])
    assert t.col.to_pylist() == [test_case.expected_value, test_case.valid_value]

    # Callable default case
    def default_value():
        return test_case.default

    class MyTable4(qv.Table):
        col = test_case.column_class(default=default_value, nullable=False, **test_case.column_kwargs)

    t = MyTable4.from_kwargs(col=[None, test_case.valid_value])
    assert t.col.to_pylist() == [test_case.expected_value, test_case.valid_value]


class TestFastCombineChunks:
    def test_no_chunks(self):
        no_chunks = pa.chunked_array([], type=pa.int64())

        have = quivr.columns._fast_combine_chunks(no_chunks)
        want = pa.array([], type=pa.int64())

        assert have.equals(want)

    def test_one_chunk(self):
        one_chunk = pa.chunked_array([[1, 2, 3]], type=pa.int64())

        have = quivr.columns._fast_combine_chunks(one_chunk)
        want = pa.array([1, 2, 3], type=pa.int64())

        assert have.equals(want)

    def test_two_chunks(self):
        two_chunks = pa.chunked_array([[1, 2, 3], [4, 5, 6]], type=pa.int64())

        have = quivr.columns._fast_combine_chunks(two_chunks)
        want = pa.array([1, 2, 3, 4, 5, 6], type=pa.int64())

        assert have.equals(want)
