from main import qa_chain

query = "what services providing by xevensolutions"
response = qa_chain.invoke({"query":query})

print(response)