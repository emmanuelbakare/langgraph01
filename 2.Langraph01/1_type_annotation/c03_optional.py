from typing import Optional

def nice_message(name: Optional[str])-> None:
    if name is  None:
        print("Hello Random Person")
    else:
        print(f"Hi there, {name}!")