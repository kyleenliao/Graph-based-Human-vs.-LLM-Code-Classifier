# DAHA GPT: Detecting AI in Homework Assignments using Graph Neural Networks

Detecting AI in homework assignments is slightly tougher than testing for AI-generated code in general. Human code often has very human hallmarks: colloquial variable names, slight redundancies, commented-out code, etc. that allow models to better distinguish the code. 

However, homework assignments are different. In the constrained environment of a CS class in high school or college, where function parameters and variable names of the code is outlined for you, it becomes harder to distinguish code. Our project is able to use a structural representation of code via Abstract Syntax Trees to distinguish human and AI-generated code with 96% accuracy!

More details about our coding process can be found here: https://medium.com/@alexandraskim/daha-gpt-detecting-ai-in-homework-assignments-using-graph-neural-networks-38a44a08d502

We used code samples from the HMCorp dataset from Xu et al. (2024), and the generated graphs can be found here! https://drive.google.com/drive/folders/1oymtRBUOBOrm6DHBWugQBzs-_sG78vge?usp=sharing

Replicate our results by running the corresponding classifier notebook for the model or with the ablation studies! To run this, you can use miniconda and install environment.yml. Download the datasets valid_no_comment.jsonl and test_no_comment.jsonl from the google drive to generate your datasets -- more information is discussed in the ablation studies notebook. 