# the annotated type is a way of attaching metadata to a type. 
# it doesnt affect the type it self but it give the user information about the type

from typing import Annotated

email = Annotated[str,'ensure you use email formt']
print(email.__metadata__) # using  __metada__  helps see the info attache to string email