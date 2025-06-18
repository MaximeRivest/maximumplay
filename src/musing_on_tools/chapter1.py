#%% [markdown]
#
# 
#
# We all have the same internet.
# Thus, fondamentally a pretrained model on 'the internet' will largely have shared broad struck capabilities.
# One obvious such capability is producing html, and production markdown. Also writing code in the common languages found both in blog posts and in code repositories.
#
# there are lots of ways we can train AI to generate output which can be parsed and run by a computer. Effectively giving it the ability to 'call' tools.
# There is a 'ranking' of how easily this can be done from existing pretrained models. I don't the what we now call 'tool calling' is the best way to do this.
#
#
# While tool calling was not invented by OpenAI, they are most likely to blame for what we call tool calling today.
# While OpenAI popularized the use of large language models (LLMs) as tool-using agents, the concept predates their work. For example, research such as "Code as Policies" demonstrated LLMs generating robot control code from natural language commands [arXiv:2209.07753](https://arxiv.org/abs/2209.07753). Similarly, "Voyager" introduced an LLM-powered agent capable of lifelong learning and tool use in Minecraft [arXiv:2305.16291](https://arxiv.org/abs/2305.16291).
# Function calling was first introduced into large language models by OpenAI on june 13th 2023 ([source](https://openai.com/index/function-calling-and-other-api-updates/))
# then in May 30th 2024 anthropic announced that Claude could now use tools.[](https://www.anthropic.com/news/tool-use-ga)
# In the open source world, llama 3.1 release on July 23rd 2024 also released a model that was further trained to use tools with this json approach (https://ai.meta.com/blog/meta-llama-3-1/).
# It was initially done as: "[...] a new way to more reliably connect GPT's capabilities with external tools and APIs."
# And at it's core it is simply that openai further trained gpt-4-0613 and gpt-3.5-turbo-0613
# To: "intelligently choose to output a JSON object containing arguments to call [...] functions"
# 
#
#https://x.com/skylar_b_payne/status/1934585851901772215
#
#https://x.com/MaximeRivest/status/1931001857624699074
#
