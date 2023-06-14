import enum

import quivr


class Color(enum.Enum):
    red = "red"
    green = "green"
    blue = "blue"


class Shoes(quivr.Table):
    size = quivr.Int64Column()
    color = quivr.EnumColumn(Color)


def main():
    # You can construct from explicit strings:
    shoes = Shoes.from_data(size=[10, 11, 12], color=["red", "green", "blue"])

    # Or from enum values:
    shoes = Shoes.from_data(size=[10, 11, 12], color=[Color.red, Color.green, Color.blue])

    print(shoes)


if __name__ == "__main__":
    main()
