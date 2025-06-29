
from typing import TypedDict

#TypeDict is a solution to python dictionary so that a dictionary items can have types
# Advantage: it ensure type saftey and the readability of the code is better
class Movie:
    name: str
    year: int 

movie = Movie(name="Good Luck", year=2022)
