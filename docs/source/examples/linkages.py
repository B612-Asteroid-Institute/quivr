from quivr import Table, StringColumn, UInt32Column, Linkage, concatenate

class People(Table):
    id = UInt32Column()
    name = StringColumn()
    age = UInt32Column()

class Pets(Table):
    name = StringColumn()
    owner_id = UInt32Column()
    species = StringColumn()

# L13

class PetsAndOwners(Table):
    owner = People.as_column()
    pets = Pets.as_column()

# L19

people = People.from_data(
    id=[1, 2, 3, 4, 5],
    name=['Bob', 'Sue', 'Joe', 'Mary', 'John'],
    age=[30, 25, 40, 35, 50]
)

pets = Pets.from_data(
    name=['Fido', 'Spot', 'Mittens', 'Rover', 'Lucy', 'Whiskers', 'Max'],
    owner_id=[1, 1, 2, 3, 4, 5, 5],
    species=['Dog', 'Dog', 'Cat', 'Dog', 'Dog', 'Cat', 'Dog']
)

linkage = Linkage(people, pets, people.id, pets.owner_id)

# L35

person, pets = linkage.select(1)
print(person.name)
# [
#  "Bob"
# ]
print(pets.name)
# [
#  "Fido",
#  "Spot"
# ]

# L48

for id, owner, pets in linkage:
    print(f"{owner.name[0]} has {len(pets)} pets")
# Mary has 1 pets
# John has 2 pets
# Bob has 2 pets
# Sue has 1 pets
# Joe has 1 pets

# L58

cat_owners = []
dog_owners = []
for id, owner, pets in linkage:
    if 'Cat' in pets.species.tolist():
        cat_owners.append(owner)
    if 'Dog' in pets.species.tolist():
        dog_owners.append(owner)

cat_owners = concatenate(cat_owners)
dog_owners = concatenate(dog_owners)

print(cat_owners.age.to_numpy().mean())
# 37.5
print(dog_owners.age.to_numpy().mean())
# 38.75

# L76
