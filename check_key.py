from dotenv import load_dotenv
import anthropic

load_dotenv()                      # reads .env and puts ANTHROPIC_API_KEY in the environment
client = anthropic.Anthropic()     # finds ANTHROPIC_API_KEY on its own

reply = client.messages.create(
    model="claude-haiku-5-5",      # the small, cheap model: right for a test
    max_tokens=50,
    messages=[{"role": "user", "content": "Reply with the single word: OK"}],
)
print(reply.content[0].text)
print("tokens used:", reply.usage.input_tokens, "in,", reply.usage.output_tokens, "out")