# Launching a RAG agent in 15 minutes with Goodmem

<style>
.rag_knowledge {
  background: #e9f9ee;
  border-left: 4px solid #27ae60;
  padding: 12px;
  border-radius: 6px;
  margin: 8px 0;
}
</style>

## Overview

<div class="rag_knowledge">
<b>What is RAG?</b> RAG is an effective way to steer LLMs to think or act per your expectation by providing them with knowledge/information that has not been "seen" by them before. 
For example, while it is impossible for a human support engineer to memorize the manual of every product sold by a company,
a RAG-based customer service chatbot can answer any question about any product as long as the information needed, even if connecting the dots is needed, is documented in the manuals.
Additionally, the answer can be up-to-date effortlessly even if the product manuals are updated frequently.
</div>

Goodmem is framework for you to build, evaluate, and optimize RAG agents. 
In this 15-minute tutorial, we will see how to build a RAG agent in Goodmem and why Goodmem is superior to other RAG frameworks in building scalable and enterprise-grade RAG agents.

## Before we start

1. Obtain an OpenAI API key (denoted as `$OPENAI_API_KEY`). Optionally, obtain a Voyager API key (denoted as `$VOYAGER_API_KEY`). Then set the environment variables:
   ```bash
   export OPENAI_API_KEY="{your_openai_api_key}"
   export VOYAGER_API_KEY="{your_voyager_api_key}" # optional
   ```
2. Install Goodmem: 
   ```bash
   curl -s "https://get.goodmem.ai" | bash
   ```
   [TODO: add flag for unattended install]
   The installation script will print out the path to Goodmem's REST API endpoint and an Goodmem API key. Be sure to write them down as we will use them in this tutorial. For convenience, export them to your shell environment for easier use later:
   ```bash
   export GOODMEM_BASE_URL="{your_goodmem_base_url}"
   export GOODMEM_API_KEY="{your_goodmem_api_key}"
   ```

3. Install the SDK depending how you'll interface with Goodmem.
   * For CLI, cURL, or Python in `requests` library, no need to install anything.
   * For using Goodmem Python SDK:
    ```bash
    pip install goodmem-client
    ```
    * [TODO] Add installation instructions for other SDKs.




## Step 1: Register a minimal RAG stack: an embedder and an LLM

Goodmem makes building RAG agents fast and scalable by adopting a declarative approach: 
1. Once RAG components (e.g., embedders, LLMs) are registered, Goodmem takes care of their wiring automatically. Just like ordering a sandwich: just say what bread and meat you want, and leave the rest to the chef. Whereas in other RAG frameworks, you need to additionally manually wire them together to ensure that data flows through them properly.
2. A RAG component can be (re)used across agents by referring to its unique ID (UUID). You do not duplicate the RAG components that would create a spaghetti system when scaling up the number of RAG agents in other frameworks.

<div class="rag_knowledge">
A <strong>minimal RAG agent</strong> consists of an embedder (converts text to embeddings to enable retrieval) and an LLM (generates the final answer using retrieved knowledge).
</div>

In this tutorial, we will use OpenAI's `text-embedding-3-small` embedder and OpenAI's `gpt-5-nano` LLM because of the popularity and high quality of OpenAI models. 

If you wanna know how to choose the embedder and the LLM for your application, you may consider Goodmem Cloud Tuner that automatically does it for you. 

### Step 1.1: Register the embedder

The three commands below, which use environment variables defined earlier, register the OpenAI `text-embedding-3-small` embedder in Goodmem and save the embedder ID returned from Goodmem to the environment variable `$EMBEDDER_ID`.

```bash
curl -X POST "$GOODMEM_BASE_URL/embedders" \
    -H "x-api-key: $GOODMEM_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "displayName": "OpenAI test",
        "providerType": "OPENAI",
        "endpointUrl": "https://api.openai.com/v1",
        "modelIdentifier": "text-embedding-3-large",
        "credentials": {
            "kind": "CREDENTIAL_KIND_API_KEY",
            "apiKey": {
                "inlineSecret": "'"$OPENAI_API_KEY"'"
            }
        }, 
        "dimensionality": 1536,
        "distributionType": "DENSE"
    }' | tee /tmp/embedder_response.json | jq
export EMBEDDER_ID=$(jq -r '.embedderId' /tmp/embedder_response.json)
echo "Embedder ID: $EMBEDDER_ID"
```
<details>
<summary>The `curl` command has very self-explanatory argument names. (Click for breakdown)</summary>

The command above uses three environment variables defined earlier: `$GOODMEM_BASE_URL`, `$GOODMEM_API_KEY`, and `$OPENAI_API_KEY`.
In this command, the embedder's display name is set as "OpenAI small". Then we tell Goodmem that this embedder's `providerType` is `OPENAI`, the `endpointUrl` is `https://api.openai.com/v1`, and the model name (`modelIdentifier`) is `text-embedding-3-small`. The credential for accessing OpenAI API is also provided in the `credentials` field which specifies the credential type (`kind`) as `CREDENTIAL_KIND_API_KEY` and the API KEY (`inlineSecret`) as `${OPENAI_API_KEY}`. Finally, we tell Goodmem that this embedder produces dense embeddings of 1536 dimensions via the `distributionType` and `dimensionality` fields. Because different embedding models may produce embeddings of different dimensions and distribution types (dense vs. sparse), it is important to specify these two fields for Goodmem to store the embeddings properly.
</details>


If the curl command above executes successfully, you will see this line at the bottom of the terminal:
```bash
Embedder ID: 019b902d-8f33-76fe-a400-e8e8caaf2e1d
```

<details>
<summary>Otherwise, an error message will be printed (click to expand)</summary>

A common error is due to duplicating an existing embedder, like this:

```json
{
  "error": "An embedder with the same provider, endpoint, model, dimensionality, distribution, and credentials already exists for this owner",
  "status": 409,
  "timestamp": 1767650962231
}
```

</details>

## Step 1.2: Register the LLM

The three commands below, which use environment variables defined earlier, register the OpenAI `gpt-5-nano` LLM in Goodmem and save the LLM ID returned from Goodmem to the environment variable `$LLM_ID`.

```bash
curl -X POST "${GOODMEM_BASE_URL}/llms" \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${GOODMEM_API_KEY}" \
  -d '{
  "displayName": "My GPT-5-nano",
  "providerType": "OPENAI",
  "endpointUrl": "https://api.openai.com/v1",
  "modelIdentifier": "gpt-5-nano",
  "credentials": {
    "kind": "CREDENTIAL_KIND_API_KEY",
    "apiKey": {
      "inlineSecret": "'"${OPENAI_API_KEY}"'"
    }
  }
}' | tee /tmp/llm_response.json | jq

export LLM_ID=$(jq -r '.llm.llmId' /tmp/llm_response.json)
echo "LLM ID: $LLM_ID"
```

<details>
<summary>The `curl` command has very self-explanatory argument names. (Click for breakdown)</summary>

The command above uses three environment variables defined earlier: `$GOODMEM_BASE_URL`, `$GOODMEM_API_KEY`, and `$OPENAI_API_KEY`.
In this command, the LLM's display name is set as "My GPT-5-nano". Then we tell Goodmem that this LLM's `providerType` is `OPENAI`, the `endpointUrl` is `https://api.openai.com/v1`, and the model name (`modelIdentifier`) is `gpt-5-nano`. The credential for accessing OpenAI API is also provided in the `credentials` field which specifies the credential type (`kind`) as `CREDENTIAL_KIND_API_KEY` and the API KEY (`inlineSecret`) as `${OPENAI_API_KEY}`. Because LLM's output are always text, there is not other mandatory fields to specify for LLM registration.
</details>

If the curl command above executes successfully, you will see this line at the bottom of the terminal:
```bash
LLM ID: 019b9071-43a9-730c-b1ed-bd3c3d38400c
```

<details>
<summary>Otherwise, an error message will be printed (click to expand)</summary>

Similar to the case for embedders, a common error is due to duplicating an existing LLM, like this:

```json
{
  "error": "An LLM with the same provider, endpoint, model, and credentials already exists for this owner",
  "status": 409,
  "timestamp": 1767655195851
}
```

</details>

## Step 2: Add knowledge for RAG agent to use

Just like we store knowledge in our brain memory, a RAG agent's knowledge is stored as **memories** in Goodmem. 
A memory can be a PDF document, a block of text (e.g., a conversation or an email), etc. 
To help management, you can group memories into **spaces**, which is also called **corpora** (the singular form is **corpus**) conventionally.
You can dynamically control the spaces that an agent can tap into -- and do it effortlessly. In contrast, in other frameworks you have to manually compile the search results from multiple spaces into one list.

The process of adding memories into spaces is called **ingestion**. 
To ingest, we first create a space and then add memory data into the space. 

### Step 2.1: Create a space

A space needs to be associated with at least one embedder. 
Goodmem supports using multiple embedders on one space for comprehensive retrieval which scores results using a weighted sum of scores from those embedders.

The three commands below create a space using the OpenAI's `text-embedding-3-small` embedder. 
Specifically, the environment variable `$EMBEDDER_ID` is the embedder ID registered in Step 1.1.

```bash
curl -X POST "$GOODMEM_BASE_URL/spaces" \
    -H "x-api-key: $GOODMEM_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "Goodmem test", 
        "spaceEmbedders": [
          {
            "embedderId": "'"$EMBEDDER_ID"'",
            "defaultRetrievalWeight": "1.0"
          }
        ]
    }' | tee /tmp/space_response.json | jq
export SPACE_ID=$(jq -r '.spaceId' /tmp/space_response.json)
echo "Space ID: $SPACE_ID"
```

If the commands above execute successfully, you will see a line like this at the bottom of the terminal:
```bash
Space ID: 019b909f-95fd-7542-9905-9fceb90145e9  
```

<details>
<summary>Otherwise, an error message will be printed (click to expand)</summary>

A common error due to the wrong embedder ID, like this:

```json
{
  "error": "Embedder not found",
  "status": 400,
  "timestamp": 1767657899350
}
```

</details>

### Step 2.2: Ingest memory data into the space

Now its time to add memory data into the space.
Usually there are two kinds of memory data:
1. Plain text (e.g., a conversation, a message, or a block of code)
2. Files (e.g., PDFs, text files, etc.)

In this demo, we will ingest both kinds of memory data into the space. In particular, for ingesting files, we will talk about **chunking**. 

#### Step 2.2.1: Ingest a plain text

The example below ingests a plain text (in field `originalContent`) into a space (in field `spaceId` which is set to the environment variable `$SPACE_ID` returned from creating a space in Step 2.1). 

```bash
curl -X POST "$GOODMEM_BASE_URL/memories" \
    -H "x-api-key: $GOODMEM_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "spaceId":"'"$SPACE_ID"'",
        "originalContent": "'"Transformers are a type of neural network architecture that are particularly well-suited for natural language processing tasks. A Transformer model leverages the attention mechanism to capture long-range dependencies in the input sequence."'",
        "contentType": "text/plain"
    }' | tee /tmp/memory_text_response.json | jq

export MEMORY_ID=$(jq -r '.memoryId' /tmp/memory_text_response.json)
echo "Memory ID: $MEMORY_ID"
```

If the commands above execute successfully, you will see a line like this at the bottom of the terminal:

```json
Memory ID: 019b90d8-2599-7521-9a4d-73c91a104e98
```

The environment variable `$MEMORY_ID` is the ID of the memory we just ingested.

<details> 
<summary>You probably also notice a line from the print out `"processingStatus": "PENDING"`.  Here is why (click to expand)</summary>

The status is `PENDING` because the ingestion is not yet complete. 

You can query the status of the ingestion by:
```bash
curl -X GET "$GOODMEM_BASE_URL/memories/$MEMORY_ID" \
    -H "x-api-key: $GOODMEM_API_KEY" \
    -H "Content-Type: application/json" | jq
```

If the commands above execute successfully, you will see something like this:
```json
{
  "memoryId": "019b90d8-2599-7521-9a4d-73c91a104e98",
  "spaceId": "019b90cb-598a-70ac-adb0-a8b69de6f90d",
  "originalContentLength": 34,
  "originalContentSha256": "a0c22c8fe43f2e369ba2399ed229a76011aa1f08e644555586356f55e8d24ec7",
  "originalContentRef": "",
  "contentType": "text/plain",
  "processingStatus": "COMPLETED",
  "metadata": {},
  "createdAt": 1767661643162,
  "updatedAt": 1767661646884,
  "createdById": "019b4c90-590b-74a0-b0bb-d67b5fe3dad4",
  "updatedById": "019b4c90-590b-74a0-b0bb-d67b5fe3dad4"
}
```

Now `processingStatus` is `COMPLETED` because the ingestion is complete. If the ingestion fails, you will see `processingStatus` to be `FAILED`. 

</details>

####  Step 2.2.3 Ingest a PDF file with chunking

A PDF file is usually text-heavy. If we treat it as one memory, such a lengthy memory will be ineffective for retrieval and inefficient or even impossible (out of context window) for the LLM to process.

<div class="rag_knowledge">
One common solution is to **chunk** the lenghty text into smaller pieces that are more manageable.
There are many chunking strategies. We provide a [guide](https://docs.goodmem.ai/docs/how-to/optimize-document-ingestion/) to help you choose the best one for your use case.
</div>

Here we will use this chunking strategy found useful in our practice:
1. Recursive: try paragraphs first, then sentences, then words.
2. Each chunk is up to 512 characters and two chunks are overlapped by 64 characters.
3. Keep the separators (such as \n\n, \n, ., etc.) at the end of the chunk. This can be expressed as the following JSON:
```json
{
  "recursive": {
    "chunkSize": 512,
    "chunkOverlap": 64,
    "keepStrategy": "KEEP_END",
    "lengthMeasurement": "CHARACTER_COUNT"
  }
}
```

In Goodmem, you don't have to handle chunking manually. Just specify the chunking strategy when uploading your PDF file and Goodmem will handle the rest.

We will use the sample file `sample_documents/employee_handbook.pdf` to illustrate. 

First, we encode the PDF file to base64.

```bash
base64 -w 0 sample_documents/employee_handbook.pdf > /tmp/file_base64.txt
```

The base64 string is very long and unfit for the `curl` command. So we buffer the JSON payload including the base64 string to a file which can be sent in the `curl` command.

```bash
jq -n \
    --arg spaceId "$SPACE_ID" \
    --rawfile originalContentB64 /tmp/file_base64.txt \
    --arg contentType "application/pdf" \
    '{
        "spaceId": $spaceId,
        "originalContentB64": $originalContentB64,
        "contentType": $contentType,
        "chunkingConfig": {
            "recursive": {
                "chunkSize": "512",
                "chunkOverlap": "64",
                "keepStrategy": "KEEP_END",
                "lengthMeasurement": "CHARACTER_COUNT"
            }
        }
    }' > /tmp/memory_request.json
```

Finally, we ingest the PDF file into the space using the buffered JSON payload:
```bash
curl -X POST "$GOODMEM_BASE_URL/memories" \
    -H "x-api-key: $GOODMEM_API_KEY" \
    -H "Content-Type: application/json" \
    --data-binary @/tmp/memory_request.json | tee /tmp/memory_response.json | jq

export MEMORY_ID=$(jq -r '.memoryId' /tmp/memory_response.json)
echo "Memory ID: $MEMORY_ID"
```

If the commands above execute successfully, you will see a line like this at the bottom of the terminal:
```bash
Memory ID: 019b90ea-6dc7-76df-8ecb-291a9cc5498d
```

You can then check the status as you did above for the plain text memory.

## Step 3: Spin the RAG agent 

Spin a RAG agent now is as easy as just one command! (Again, we added the `jq` command which is not necessary but nice to have to parse the response.)

```bash
curl -X POST "$GOODMEM_BASE_URL/memories:retrieve" \
    -H "x-api-key: $GOODMEM_API_KEY" \
    -H "Content-Type: application/json" \
    -H "Accept: application/x-ndjson" \
    -d '{
        "message": "artificial intelligence",
        "spaceKeys": [
            {
                "spaceId": "'"$SPACE_ID"'"
            }
        ], 
        "requestedSize": 1
    }' | jq -r 'select(.retrievedItem) | {
        memoryId: .retrievedItem.chunk.chunk.memoryId,
        chunkText: .retrievedItem.chunk.chunk.chunkText,
        relevanceScore: .retrievedItem.chunk.relevanceScore
    }'
```

Basically, we ask the RAG agent to answer the query "artificial intelligence" (`message="artificial intelligence"`) and retrieve the most relevant (`requestedSize=1`) memory chunk. The expected response is:
```json
{
  "memoryId": "019b910e-dfab-76df-bfb7-5896a234158f",
  "chunkText": "Transformers are a type of neural network architecture that are particularly well-suited for natural language processing tasks. A Transformer model leverages the attention mechanism to capture long-range dependencies in the input sequence.",
  "relevanceScore": -0.24954041838645935
}
```

Do you recall what we ingested into the space? This is indeed the most matching memory chunk! Other chunks are employee policies, etc.

You probably noticed that the command contains a list of space IDs. This is a feature of Goodmem that allows you to freely and dynamically mix spaces, meaning your RAG agent is fluid! Again this is the benefit of giving every component a unique ID.

TODO: Add LLM. 

## Step 4 (optional): Add a reranker

## Step 5 (advanced): Get the most out of your RAG stack using Goodmem Cloud Tuner

This is a feature only for paid customers. Please contact our sales team for more details.

THE END.



TODO: 
1. The example here https://docs.goodmem.ai/docs/how-to/optimize-document-ingestion/#comparison-examples misses many mandatory fields.
2. None is a bad default chunking strategy. 
3. No default value for optional arguments here https://docs.goodmem.ai/docs/reference/api-reference/rest/memories/retrieveMemory/
4. The camelCase vs. snake_case issue.

```bash
# tmp code
export GOODMEM_BASE_URL="http://localhost:8080/v1"
export GOODMEM_API_KEY="gm_twjgllyzsfh7tzsffqtog4pbnm"
```