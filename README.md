Task 1: News Topic Classifier Using BERT

Problem Statement 
1.What is the problem? News websites publish thousands of articles every day. Each article needs to be sorted into a topic (Sports, Tech, Politics, etc.) so readers can find what they want. Doing this manually is slow.
2.Our Goal: Train an AI model that reads a news headline and automatically classifies it into one of 4 categories:
World → international news
Sports → sports news
Business → business/finance news
Sci/Tech → science and technology news
3.What is BERT? BERT (Bidirectional Encoder Representations from Transformers) is a powerful pre-trained language model made by Google. It already understands English very well. We just need to fine-tune it (teach it our specific task).
4.What is Fine-Tuning? BERT is like a student who already knows English perfectly. Fine-tuning is like teaching that student one specific skill: identifying news topics. We only need a little training!
5.Dataset: AG News Dataset (available on Hugging Face)

Objective
Fine-tune the BERT transformer model to classify news headlines into 4 categories: World, Sports, Business, and Sci/Tech. This demonstrates transfer learning by using a pre-trained model and adapting it to a specific task.
Dataset
•Name: AG News Dataset
•Source: Hugging Face Datasets (loaded automatically)
•Size: 120,000 training samples, 7,600 test samples
•Categories: World, Sports, Business, Sci/Tech (perfectly balanced)
Methodology / Approach
Step 1: Data Loading & Exploration
•Loaded AG News dataset from Hugging Face
•Visualized category distribution: all 4 classes are balanced
•Used a small subset (2000 train, 500 test) for fast training
Step 2: Tokenization
•Loaded bert-base-uncased tokenizer
•Converted text to token IDs with max length 128
•Applied padding and truncation
Step 3: Fine-Tuning
•Loaded bert-base-uncased with 4 classification outputs
•Used Hugging Face Trainer API for automatic training loop
•Trained for 3 epochs with warmup steps and weight decay
Step 4: Evaluation
•Accuracy and F1 Score calculated on test set
•Confusion matrix plotted for detailed class-by-class analysis
Step 5: Deployment
•Built interactive Gradio web demo
•User can type any headline and get instant category prediction with confidence scores
Key Results
Metric	Value
Accuracy	~87%
F1 Score (weighted)	~85-87%
Model	bert-base-uncased
Training Samples	2,000
Epochs	3
Deployment	Gradio Web App

•Sports and Business are easiest to classify due to distinct vocabulary
•World and Sci/Tech occasionally overlap in technology related global news
•Transfer learning is highly effective even with small training sets
Conclusion
BERT achieved ~87% accuracy on news classification with only 2000 training samples. This demonstrates the power of transfer learning. The Gradio deployment makes it  shareable as a web application without any web development knowledge.


Task 2: End-to-End ML Pipeline (Customer Churn Prediction)
Problem Statement 
1.What is Customer Churn? Churn means a customer leaving a company (cancelling their subscription). Telecom companies lose a lot of money when customers leave.
2.Our Goal: Build a complete, reusable ML Pipeline that predicts:
Output “1”→ Customer WILL leave (churn)
Output “0”→ Customer will STAY
3.What is an ML Pipeline? Instead of doing steps one by one manually, a Pipeline performs all steps together automatically: Raw Data → Clean → Scale → Encode → Train Model → Predict
4.Why is this useful? Once built, you can feed any new customer data into it and get a prediction automatically.
Objective
To build a complete, reusable, and production ready ML pipeline that predicts whether a telecom customer will leave or not. The pipeline chains all preprocessing and modeling steps together automatically.
Dataset
•Name: Telco Customer Churn Dataset
•Source: WA_Fn-UseC_-Telco-Customer-Churn.csv
•Size: 7,043 customer records, 21 features
•Target: Churn (Yes=1 = customer left, No=0 = customer stayed)
Methodology / Approach
Step 1: Data Cleaning
•Removed customerID (no predictive value)
•Fixed Total Charges column (stored as text, converted to number)
•Converted Churn from Yes/No to 1/0
Step 2: Pipeline Construction
•Numerical pipeline: SimpleImputer (median) + StandardScaler
•Categorical pipeline: SimpleImputer (most frequent) + OneHotEncoder
•Combined using Column Transformer: applies right pipeline to right columns
Step 3: Model Training
•Logistic Regression pipeline: preprocessor + LogisticRegression
•Random Forest pipeline: preprocessor + RandomForestClassifier (100 trees)
•80/20 train test split with random_state=42
Step 4: GridSearchCV Hyperparameter Tuning
•Tuned: n_estimators, max_depth, min_samples_split
•8 combinations x 3-fold CV = 24 training runs
•Best settings automatically selected
Step 5: Export
•Complete pipeline saved to churn_pipeline.pkl using joblib
•Can be loaded and used for predictions without retraining
Key Results
Model	Accuracy
Logistic Regression	78.75%
Random Forest	77.75%
Tuned Random Forest (Best)	80.44%
A complete, reusable ML Pipeline
Handles missing values, scaling, and encoding automatically
Saved to a file which means it is ready for real analysis
Can predict churn for any new customer easily
Conclusion
We built a complete end-to-end ML pipeline achieving good accuracy. The pipeline handles all preprocessing automatically and is saved to a file for future reuse. Businesses can use this to identify at-risk customers and offer them special deals to avoid leaving.


Task 5: Auto Tagging Support Tickets Using LLM
Problem Statement
1.What are Support Tickets? When a customer has a problem, they write a complaint/request to a company. Example: 'My internet is not working since yesterday!' These are called support tickets.
2.The Problem: Companies receive tons of tickets daily. Someone has to read each one and assign it a tag/category so the right team handles it. This takes a lot of time and human effort.
3.Solution: It uses large language model to automatically read each ticket and assign the top 3 most relevant tags. 
4.Model Used: Hugging Face free API (Mistral-7B-Instruct). Three Techniques We Will Compare:
1.Zero-Shot: Give AI no examples, just ask it to tag directly.
2.Few-Shot: Give AI 2-3 example tickets with their tags.
3.Chain-of-Thought: Ask the AI to think step by step before tagging.
Objective
Build a system that automatically reads customer support tickets and assigns the top 3 most relevant tags using a Large Language Model and prompt engineering without any model training.
Dataset
•Custom dataset of 10 realistic support tickets
•10 available tag categories: Billing Issue, Technical Problem, Account Access, etc.
•Each ticket has correct tags for accuracy evaluation
Methodology / Approach
Method 1: Zero-Shot Learning
•No examples provided, just task description and available tags
•AI uses pre-trained knowledge to assign tags directly
•Fastest method but least accurate
Method 2: Few-Shot Learning
•3 example tickets with correct tags shown before the question
•AI learns the expected format and logic from examples
•Significantly improves accuracy over zero-shot
Method 3: Chain-of-Thought Reasoning
•AI asked to think step-by-step before giving final answer
•Steps: identify problem, identify department, find secondary issues, assign tags
•Most accurate method which reduces mistakes



Key Results
Method	How It Works	Speed
Zero-Shot	No examples, direct question	Fastest
Few-Shot	3 examples provided	Medium
Chain-of-Thought	Step-by-step reasoning	Slowest

•Prompt engineering alone solves real business problems without model training
•Temperature=0.3 gives more consistent, focused answers
•Few-Shot and Chain-of-Thought significantly outperform Zero-Shot
Conclusion
We built an LLM-powered auto-tagging system using three prompt engineering techniques. Chain-of-Thought prompting achieved the best accuracy by asking the model to reason before answering. This demonstrates that clever prompt design alone without fine-tuning can solve real business problems effectively and cheaply.
