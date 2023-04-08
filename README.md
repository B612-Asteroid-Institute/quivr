# quiver

Quiver is a Python library which provides great containers for Arrow data.

Quiver's `Table`s are like DataFrames, but with strict schemas to
enforce types and expectations. They are backed by the
high-performance Arrow memory model, making them well-suited for
streaming IO, RPCs, and serialization/deserialization to Parquet.

## why?

Data engineering involves taking analysis code and algorithms which
were prototyped, often on pandas DataFrames, and shoring them up for
production use.

While DataFrames are great for ad-hoc exploration, visualization, and
prototyping, they aren't as great for building sturdy applications:

 - Loose and dynamic typing makes it difficult to be sure that code is
   correct without lots of explicit checks of the dataframe's state.
 - Performance of Pandas operations can be unpredictable and have
   surprising characteristics, which makes it harder to provision
   resources.
 - DataFrames can use an extremely large amount of memory (typical
   numbers cited are between 2x and 10x the "raw" data's size), and
   often are forced to copy data in intermediate computations, which
   poses unnecessarily heavy requirements.
 - The mutability of DataFrames can make debugging difficult and lead
   to confusing state.
   
We don't want to throw everything out, here. Vectorized computations
are often absolutely necessary for data work. But what if we could
have those vectorized computations, but with:
 - Types enforced at runtime, with no dynamically column information.
 - Relatively uniform performance due to a no-copy orientation
 - Immutable data, allowing multiple views at very fast speed
  
This is what Quiver's Tables try to provide.

## Installation

Check out this repo, and `pip install` it.

## Usage

Your main entrypoint to Quiver is through defining classes which
represent your tables. You write a `pyarrow.Schema` as the `schema`
class attribute of your class, and Quiver will take care of the rest.

```python
from quiver import TableBase
import pyarrow as pa


class People(TableBase):
    schema = pa.schema(
	    [
	        pa.field("name", pa.string()),
			pa.field("height_cm", pa.float64()),
			pa.field("weight_kg", pa.float64()),
			pa.field("birthdate", pa.date64()),
	    ]
	)
```







