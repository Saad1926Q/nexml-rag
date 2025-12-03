from supermemory import Supermemory
import os
from dotenv import load_dotenv
load_dotenv()
SUPERMEMORY_API_KEY = os.getenv('SUPERMEMORY_API_KEY')
client = Supermemory(
    api_key=SUPERMEMORY_API_KEY,
)

response = client.memories.add(
    content="SuperMemory Python SDK is awesome.",
    container_tag="Python_SDK",
    metadata={
        "note_id": "123",
    }
)
print(response)

searching = client.search.execute(
    q="What do you know about me?",
)
print(searching.results)