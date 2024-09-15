PRIMING = u"""
Given the intention:

{{intention}}

and an utterance:

{{utterance}}

Generate a 6-turn conversation between a user and a chatbot.

"""

KNOWLEDGE_INJECTION = u"""
Given the conversation between a user and chatbot:

{{conversation}}

Extend the previous conversation between a user and chatbot for 6 more turns that use the following knowledge:

{{knowledge}}

"""








