## **Managing chat history**

### **Table of Contents**

- [Description](#description)
- [Useful Notes](#useful-notes)
- [Development Steps](#development-steps)
- [Deliverables](#deliverables)
- [Useful Resources](#useful-resources)
    - [Topics and Projects](#topics)
    - [Docs](#docs)

---

### Description

Storing chat history is important as it ensures that the model has context for coherent, on-topic responses. There are various strategies you can use to store chat history for your LangChain apps, like in-memory, local file storage, or in a database such as Redis or SQLite.

You need to be careful because chat history can become too long, leading to token misuse. For example, imagine a case where your chat history spans days and is becoming too long, even when the oldest messages aren’t needed.

In this task, we look at strategies to manage and trim chat history so that your models always have the right balance between context and token usage.

---

### Useful Notes

As you already know, LangChain provides several integrations you can use in your LLM applications. One such integration is with Redis. **Redis** is an open-source, in-memory key-value store used as a database, cache, and message broker, offering microsecond-level performance. It offers features like persistence, replication, and clustering.

To use Redis, you can use LangChain’s Redis integration, which allows you to use Redis for chat history and cache management. For now, we’ll focus on using Redis for chat history management. You can also use it for caching via LiteLLM as well.

Using Redis to store message history for your LangChain apps can be done in a few steps. First, you need Redis. You can easily set it up using Docker. If there is another Redis container running, the default Redis port might already be in use. Therefore, you’d need to map it to a different host port (the syntax is `host:container` port):

```bash
docker run --restart always --name hyper-redis -d -p 6380:6379 redis redis-server --save 60 1
```

This pulls the latest version of Redis and runs the container (`hyper-redis`) in detached mode. Now, Redis will be available at `redis://localhost:6380/0`. The `--save 60 1` option tells Redis to take a snapshot every 60 seconds if there has been at least one write operation. This snapshot is persisted to disk.

You will also need the appropriate libraries to interact with Redis:

```bash
pip install langchain-redis redis
```

To store message history in Redis for your LangChain apps, you use the [RedisChatMessageHistory](https://reference.langchain.com/python/langchain-redis/chat_message_history/RedisChatMessageHistory) class. Simply import the class and initialize chat history as follows:

```python
from langchain_redis import RedisChatMessageHistory

REDIS_URL = "redis://localhost:6380/0"

history = RedisChatMessageHistory(session_id = "your-session-id", redis_url=REDIS_URL)
```

Next, you can add messages:

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
system_prompt = "You are a helpful assistant."

user_input = input().strip()

history.add_message(SystemMessage(content=system_prompt)) # to add a system message (you may not need to add this to the chat history)
history.add_message(HumanMessage(content=user_input)) # to add a user's input
response = llm.invoke(....)
history.add_message(AIMessage(content=response.content)) # to add an AI response
```

You can then retrieve the messages as follows:

```python
for message in history.messages:
    print(f"{message.type}: {message.content}")
    
# output
SystemMessage: You are a helpful assistant. 
HumanMessage: I need help ordering a smartphone.
AIMessage: Hello! I can assist you with smartphone recommendations
```

You could even search for specific messages in the chat history. This is particularly useful when you need to only send a subset of messages to the LLM rather than the entire chat history to save costs. You can search for a message as follows:

```python
search_results = history.search_messages("smartphone")
for result in search_results:
    print(f"{result['type']}: {result['content'][:10]}")   
    
# Output
HumanMessage: I need help ordering a smartphone.
AIMessage: Hello! I can assist you with smartphone recommendations
```

To prevent the chat history from getting too long, you can clear it periodically (for example, via a helper function that deletes messages older than five days):

```python
history.clear()
```

If you are using LangChain chains, you can manage chat history using Redis instead of a local variable. The prompts remain the same:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        langfuse_prompt.get_langchain_prompt()[0],
        MessagesPlaceholder(variable_name="conversation"),
    ]
)
```

There are two approaches to integrating Redis chat history:

### Option 1: Using `RunnableWithMessageHistory` (Automated)

This approach automatically saves and loads all messages through the chain:

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_redis import RedisChatMessageHistory

def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(session_id, redis_url=REDIS_URL)

chain = prompt | llm
chain_with_message_history = RunnableWithMessageHistory(
    chain, get_redis_history, input_messages_key="user_input", history_messages_key="conversation"
)

response = chain_with_message_history.invoke(
    {"user_input": "Hey. I'm from New York."},
    config={"configurable": {"session_id": "hyperskill_user"}},
)
```

**Limitation**: This saves all messages that pass through the chain, including tool calls, intermediate AI messages, and tool results. This increases storage and token costs.

### Option 2: Manual Management with `RedisChatMessageHistory` (Recommended for Tool-Based Applications)

For applications using tools, manual management provides control over what gets persisted:

```python
from langchain_redis import RedisChatMessageHistory

# Initialize Redis history with TTL
redis_history = RedisChatMessageHistory(
    session_id=session_id,
    redis_url=REDIS_URL,
    ttl=3600  # 1 hour
)

# Load conversation history from Redis
conversation = list(redis_history.messages)

# Add user input to in-memory conversation
user_message = HumanMessage(user_input)
conversation.append(user_message)

# Run chains (tool calls happen in-memory)
context_chain.invoke({"user_input": user_input, "conversation": conversation}, ...)
response = review_chain.invoke({"user_id": user_id, "user_input": user_input, "conversation": conversation}, ...)

# Save only user input and final response to Redis
redis_history.add_message(user_message)
redis_history.add_message(response)
```

This approach keeps conversation history clean (only user messages and final responses), reduces token usage, and provides explicit control over persistence. Tool calls and intermediate messages remain in memory for the current turn only.

**Note**: When using manual management, update the `generate_context` function to accept `conversation` as a parameter:

```python
def generate_context(ai_message: AIMessage, conversation: list) -> dict:
    """Process tool calls and add results to the conversation list"""
    conversation.append(ai_message)
    # ... rest of the function
```

Unfortunately, the chat history is bound to get too long over time. Fortunately, there are strategies you can use to ensure that this does not happen. For example, you can set a time to live (TTL) for Redis chat history:

```python
 RedisChatMessageHistory(session_id, redis_url=REDIS_URL, ttl=120)
```

Here, we are setting the duration to two minutes (120 seconds). In addition, you can implement a mechanism to trim long messages. LangChain makes this easy with `trim_messages()`. All you need to do is initialize it and pass your chat history through it. Here is an example:

```python

from langchain_core.messages import trim_messages

trimmer = trim_messages(
    strategy="last", # keep either the last or first messages
    token_counter=llm, # use your LLM to count tokens or create a special function
    max_tokens=500, # the maximum number of tokens
    start_on="human", # the first message type in the trimmed history
    end_on=("human", "tool"), # the last message type in the trimmed history
    include_system=True, # always include the system message
)

chain = prompt | trimmer | llm
```

Now, the chat history won’t get too long, potentially using up more tokens than it should. However, remember to balance cost savings with enough context.

---

### Development Steps

Replace the in-memory conversation storage with Redis-based chat history management. Follow these steps:

1. **Set up Redis**: Run the Docker command to start Redis on port 6380 (port 6379 is used by Langfuse):
   ```bash
   docker run --restart always --name hyper-redis -d -p 6380:6379 redis redis-server --save 60 1
   ```

2. **Add environment variable**: Set `REDIS_URL=redis://localhost:6380/0` in your `.env` file.

3. **Update conversation management**:
   - Remove the global `conversation = []` variable
   - Initialize `RedisChatMessageHistory` with a TTL (e.g., 3600 seconds for 1 hour)
   - At the start of each turn, load messages from Redis using `list(redis_history.messages)`
   - Save only the user's `HumanMessage` and the final `AIMessage` to Redis
   - Keep tool calls and intermediate messages in memory for the current turn only

4. **Update `generate_context` function**: Change the signature to accept `conversation` as a parameter instead of using a global variable:
   ```python
   def generate_context(ai_message: AIMessage, conversation: list) -> dict:
   ```

5. **Add message trimming to context chain**: Insert a trimmer in the context chain pipeline to limit token usage:
   ```python
   from langchain_core.messages import trim_messages

   trimmer = trim_messages(
       strategy="last",
       token_counter=llm,
       max_tokens=500,
       start_on="human",
       end_on=("human", "tool"),
       include_system=True,
   )

   context_chain = context_prompt | trimmer | llm_with_tools
   ```

The application should continue to function identically, but conversation history will now persist in Redis and automatically expire after the TTL period.

---

### Deliverables
- Implement Redis chat history management in your LangChain app.
- Test the chat history management by running a few queries and checking that the chat history is stored correctly in Redis.
- Ensure that the chat history is cleared or trimmed after a certain period to prevent it from growing too long.

--- 

### **Useful Resources**

###### **Docs**

- [Getting started with Redis](https://redis.io/docs/latest/get-started/).
- [Redis Chat Message history](https://reference.langchain.com/python/langchain-redis/chat_message_history/RedisChatMessageHistory).
- [How to trim messages](https://reference.langchain.com/python/langchain-core/messages/utils/trim_messages).

---

Next: [LLM Safety and Security: Guardrails](./task_3.md)
