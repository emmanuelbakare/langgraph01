import json
from langchain_core.messages import HumanMessage, AIMessage


data = ""

def message_to_dict(msg):
    return {
        "type": msg.__class__.__name__,
        "content": msg.content,
        "metadata": msg.__dict__.get("response_metadata", {}),
        "id": msg.__dict__.get("id", None),
    }

json_data = {
    "messages": [message_to_dict(m) for m in data["messages"]]
}

with open("output.json", "w") as f:
    json.dump(json_data, f, indent=4)