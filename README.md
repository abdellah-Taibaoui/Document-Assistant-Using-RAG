# Document Assistant Using RAG

we're going to build Local RAG pipeline :

![rag pipline](https://github.com/user-attachments/assets/31857112-fad1-4ee9-9c1f-28c6809e5f93)


In our specific example, we'll build a RAG workflow using langchain and huggingface that allows a person to query PDF version of a Medical Textbooks that contain 558 pages and have an LLM generate responses back to the query based on passages of text from the Medical textbook.

PDF source: [poly_pharmacologie_generale.pdf](https://www.pharmacobx.fr/documents/pharmacologiegenerale/poly_pharmacologie_generale.pdf).

LLM models used :[Mistral OpenOrca](https://huggingface.co/mav23/Mistral-7B-OpenOrca-GGUF) and [TinyLlama](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) 

embedding model used :[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## Getting Started

Note: Tested in Python 3.11, running on Windows 10 without GPU and on 8GB of RAM.


## Prerequisites and Setup

### Using Google Colab

You can run notebook `RAG_App.ipynb` directly in [Google Colab](https://colab.research.google.com/drive/17DVKATD3dRTBlAKbe-2CmpgQ3K4RF1zo?usp=sharing). 


### Using Docker To Do a Demo
To facilitate container deployment, I used TinyLlama, a lighter LLM model that is simpler and easier to deploy.
And i asked the model : "c'est quoi Les effets anaphylactoïdes ? "
#### Install [visual studio cpp](https://visualstudio.microsoft.com/fr/downloads/)
#### Install and run [Docker desktop](https://docs.docker.com/desktop/)

#### Download and open files to build the docker container :  [docker Files Link](https://drive.google.com/file/d/1_moMOPloKRvgo4d6xmtaaw4ff1SOL4Ki/view?usp=sharing)

#### Build the docker file to run the RAG_system
```
docker-compose up --build
```

The Demo result 

 ![docker_demo](https://github.com/user-attachments/assets/957d0b86-3342-40db-bfc9-5df6c762fb52)

### Using Jupiter Notebook

#### Clone repo

```
git clone https://github.com/abdellah-Taibaoui/RAG_Application.git
```

```
cd RAG_Application
```

### Create environment

```
python -m venv venv
```

### Activate environment

Linux/macOS:
```
source venv/bin/activate
```

Windows: 
```
.\venv\Scripts\activate
```

### Install requirements


#### Install [visual studio cpp](https://visualstudio.microsoft.com/fr/downloads/)

```
pip install -r requirements.txt
```

### Launch notebook

VS Code:

```
code app.ipynb
```

Jupyter Notebook

```
jupyter notebook app.ipynb

```


**Setup notes:** 
* If you run into any install/setup troubles, please leave an issue.
* installing and compiling can take ~5 minutes to 1 hours depending on your system setup.

## RAG pipeline Result example

![result](https://github.com/user-attachments/assets/c759d9bf-e3e7-437a-9f44-d10e83cc50c6)

* The question was : " c'est quoi Les effets anaphylactoïdes ? "

* LLM without RAG response :
  
"'\n\nComparaisons : \n1. Les effets anaphylactoïdes sont des effets secondaires de la maladie, souvent négatifs pour le patient. Ils ont tendance à surmonter ou supprimer les effets positifs d’une maladie (effet de repression). \n2. Les effets anaphylactoïdes sont généralement négatifs pour la santé publique et peuvent conduire à des déstabilisations, sauf si elle a été pratiquée avec une utilisation correcte. \n3. C’est-à-dire que les effets anaphylactoïdes sont inquiétants mais pas forcément négatifs, et leur utilisation doit être prudente. \n\nConséquences : \n1. Les effets anaphylactoïdes peuvent entraîner des maladies graves ou d’ultimes, particulièrement chez les patients qui ont recours à des traitements par voûte ou autres méthodes. \n2. Des effets anaphylactoïdes peuvent s’accommod "

* LLM with RAG response :
  
"Les effet(s) anaphylactoïdes sont des manifestations nocive(s) cliniquement procésées(s) des effets immuno-globulié(s) IgE. C'est-à-dire que ces manifestations (lesquelles peuvent être sévères) sont liées aux immuno-globulines IgE, c'est-à-dire aux médicaments anti-allergique ou au traitement immunologique."

* The correct response from the document used: 

![response_from_doc](https://github.com/user-attachments/assets/4d27d44e-6d49-42dd-9fa7-d1ddaafde51d)



## What is RAG?


RAG stands for Retrieval Augmented Generation.

It was introduced in the paper [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401).

Each step can be roughly broken down to:

* **Retrieval** - Seeking relevant information from a source given a query. For example, getting relevant passages of Wikipedia text from a database given a question.
* **Augmented** - Using the relevant retrieved information to modify an input to a generative model (e.g. an LLM).
* **Generation** - Generating an output given an input. For example, in the case of an LLM, generating a passage of text given an input prompt.

## Why RAG?

The main goal of RAG is to improve the generation outptus of LLMs.

Two primary improvements can be seen as:
1. **Preventing hallucinations** - LLMs are incredible but they are prone to potential hallucination, as in, generating something that *looks* correct but isn't. RAG pipelines can help LLMs generate more factual outputs by providing them with factual (retrieved) inputs. And even if the generated answer from a RAG pipeline doesn't seem correct, because of retrieval, you also have access to the sources where it came from.
2. **Work with custom data** - Many base LLMs are trained with internet-scale text data. This means they have a great ability to model language, however, they often lack specific knowledge. RAG systems can provide LLMs with domain-specific data such as medical information or company documentation and thus customized their outputs to suit specific use cases.

The authors of the original RAG paper mentioned above outlined these two points in their discussion.

> This work offers several positive societal benefits over previous work: the fact that it is more
strongly grounded in real factual knowledge (in this case Wikipedia) makes it “hallucinate” less
with generations that are more factual, and offers more control and interpretability. RAG could be
employed in a wide variety of scenarios with direct benefit to society, for example by endowing it
with a medical index and asking it open-domain questions on that topic, or by helping people be more
effective at their jobs.

RAG can also be a much quicker solution to implement than fine-tuning an LLM on specific data. 


## What kind of problems can RAG be used for?

RAG can help anywhere there is a specific set of information that an LLM may not have in its training data (e.g. anything not publicly accessible on the internet).

For example you could use RAG for:
* **Customer support Q&A chat** - By treating your existing customer support documentation as a resource, when a customer asks a question, you could have a system retrieve relevant documentation snippets and then have an LLM craft those snippets into an answer. Think of this as a "chatbot for your documentation". Klarna, a large financial company, [uses a system like this](https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/) to save $40M per year on customer support costs.
* **Email chain analysis** - Let's say you're an insurance company with long threads of emails between customers and insurance agents. Instead of searching through each individual email, you could retrieve relevant passages and have an LLM create strucutred outputs of insurance claims.
* **Company internal documentation chat** - If you've worked at a large company, you know how hard it can be to get an answer sometimes. Why not let a RAG system index your company information and have an LLM answer questions you may have? The benefit of RAG is that you will have references to resources to learn more if the LLM answer doesn't suffice.
* **Textbook Q&A** - Let's say you're studying for your exams and constantly flicking through a large textbook looking for answers to your quesitons. RAG can help provide answers as well as references to learn more.

All of these have the common theme of retrieving relevant resources and then presenting them in an understandable way using an LLM.

From this angle, you can consider an LLM a calculator for words.

## Why local?

Privacy, speed, cost.

Running locally means you use your own hardware.

From a privacy standpoint, this means you don't have send potentially sensitive data to an API.

From a speed standpoint, it means you won't necessarily have to wait for an API queue or downtime, if your hardware is running, the pipeline can run.

And from a cost standpoint, running on your own hardware often has a heavier starting cost but little to no costs after that.

Performance wise, LLM APIs may still perform better than an open-source model running locally on general tasks but there are more and more examples appearing of smaller, focused models outperforming larger models. 

## Key terms

| Term | Description |
| ----- | ----- | 
| **Token** | A sub-word piece of text. For example, "hello, world!" could be split into ["hello", ",", "world", "!"]. A token can be a whole word,<br> part of a word or group of punctuation characters. 1 token ~= 4 characters in English, 100 tokens ~= 75 words.<br> Text gets broken into tokens before being passed to an LLM. |
| **Embedding** | A learned numerical representation of a piece of data. For example, a sentence of text could be represented by a vector with<br> 768 values. Similar pieces of text (in meaning) will ideally have similar values. |
| **Embedding model** | A model designed to accept input data and output a numerical representation. For example, a text embedding model may take in 384 <br>tokens of text and turn it into a vector of size 768. An embedding model can and often is different to an LLM model. |
| **Similarity search/vector search** | Similarity search/vector search aims to find two vectors which are close together in high-demensional space. For example, <br>two pieces of similar text passed through an embedding model should have a high similarity score, whereas two pieces of text about<br> different topics will have a lower similarity score. Common similarity score measures are dot product and cosine similarity. |
| **Large Language Model (LLM)** | A model which has been trained to numerically represent the patterns in text. A generative LLM will continue a sequence when given a sequence. <br>For example, given a sequence of the text "hello, world!", a genertive LLM may produce "we're going to build a RAG pipeline today!".<br> This generation will be highly dependant on the training data and prompt. |
| **LLM context window** | The number of tokens a LLM can accept as input. For example, as of March 2024, GPT-4 has a default context window of 32k tokens<br> (about 96 pages of text) but can go up to 128k if needed. A recent open-source LLM from Google, Gemma (March 2024) has a context<br> window of 8,192 tokens (about 24 pages of text). A higher context window means an LLM can accept more relevant information<br> to assist with a query. For example, in a RAG pipeline, if a model has a larger context window, it can accept more reference items<br> from the retrieval system to aid with its generation. |
| **Prompt** | A common term for describing the input to a generative LLM. The idea of "[prompt engineering](https://en.wikipedia.org/wiki/Prompt_engineering)" is to structure a text-based<br> (or potentially image-based as well) input to a generative LLM in a specific way so that the generated output is ideal. This technique is<br> possible because of a LLMs capacity for in-context learning, as in, it is able to use its representation of language to breakdown <br>the prompt and recognize what a suitable output may be (note: the output of LLMs is probable, so terms like "may output" are used). | 
