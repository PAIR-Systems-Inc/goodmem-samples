# Launching a RAG agent in 5 minutes with Goodmem

## Overview

RAG is an effective way to steer LLMs to think or act per your expectation by providing them with knowledge/information that has not been "seen" by them before. 
For example, while it is impossible for a human support engineer to memorize the manual of every product sold by a company,
a RAG-based customer service chatbot can answer any question about any product as long as the information needed, even if connecting the dots is needed, is documented in the manuals.
Additionally, the answer can be up-to-date effortlessly even if the product manuals are updated frequently.

Goodmem is framework for you to build, evaluate, and optimize RAG agents. 
In this 5-minute tutorial, we will see how to build a RAG agent in Goodmem and why Goodmem is superior to other RAG frameworks in building scalable and enterprise-grade RAG agents.

## Before we start

1. Obtain an OpenAI API key (denoted as `$OPENAI_API_KEY`). Optionally, obtain a Voyager API key (denoted as `$VOYAGER_API_KEY`). 
2. Install Goodmem: 
   ```bash
   curl -s "https://get.goodmem.ai" | bash
   ```
   [TODO: add flag for unattended install]
   The installation script will give you the path to Goodmem's REST API endpoint (denoted as `$GOODMEM_API_URL`) and your Goodmem API key (denoted as `$GOODMEM_API_KEY`). Be sure to write them down as we will use them in this tutorial.
   
   You may export them to your shell environment for easier use later:
   ```bash
   export GOODMEM_BASE_URL="{your_goodmem_base_url}"
   export GOODMEM_API_KEY="{your_goodmem_api_key}"
   ```

3. Depending on your choice of interfacing with Goodmem, e.g., via CLI or Python, please install such interface. 
    For Python, please intall the Goodmem Python package:
    ```bash
    pip install goodmem-client
    ```
   

## Step 1: Register a minimal RAG stack: an embedder and an LLM

Building RAG applications in Goodmem is like no other RAG frameworks: 
* **Fast**: Just two steps: register RAG components (e.g., embedders, LLMs) and compose them into a RAG application. 
No need for passing data between them manually as they would have to do in other RAG frameworks.
* **Scalable**: each RAG component is assigned with a unique ID (UUID), allowing easy reuse and management across multiple RAG components and agents. 
In contrast, many existing RAG frameworks do not support this mechanism and duplicated RAG components cause management overhead and risk.
* Combining these two designs, Goodmem has a third advantage: One RAG can employ multiple RAG components of the same purpose, e.g., two embedders to balance different aspects of searching.
In other framework, the developer needs different code for one, two, three, etc. parallel components.  


A **minimal RAG agent** consists of two components: an embedder and a large language model (LLM).
At a high level, an embedder's job is to enable effective retrieval of the right knowledge to fulfill a user query, while an LLM's job is to fulfills the user query based on the knowledge retrieved.

In this tutorial, we will use OpenAI's `text-embedding-3-small` embedder and OpenAI's `gpt-5-nano` LLM.

### Register the embedder

The command below registers the OpenAI `text-embedding-3-small` embedder in Goodmem:

```bash
curl -X POST "${GOODMEM_BASE_URL}/embedders" \
    -H "x-api-key: ${GOODMEM_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{
        "displayName": "OpenAI small",
        "providerType": "OPENAI",
        "endpointUrl": "https://api.openai.com/v1",
        "modelIdentifier": "text-embedding-3-large",
        "dimensionality": 1536,
        "distributionType": "DENSE",
        "credentials": {
            "kind": "CREDENTIAL_KI
            "apiKey": {
            "inlineSecret": "${OPENAI_API_KEY}"
            }
        }
    }' | jq
```


<details>
<summary><strong>Expected response and analysis (click to expand)</strong></summary>

The command above uses three environment variables defined earlier: `$GOODMEM_BASE_URL`, `$GOODMEM_API_KEY`, and `$OPENAI_API_KEY`.
In this command, the embedder's display name is set as "OpenAI small". Then we tell Goodmem that this embedder's `providerType` is `OPENAI`, the `endpointUrl` is `https://api.openai.com/v1`, and the model name (`modelIdentifier`) is `text-embedding-3-small`. The credential for accessing OpenAI API is also provided in the `credentials` field which specifies the credential type (`kind`) as `CREDENTIAL_KIND_API_KEY` and the API KEY (`inlineSecret`) as `${OPENAI_API_KEY}`. Finally, we tell Goodmem that this embedder produces dense embeddings of 1536 dimensions via the `distributionType` and `dimensionality` fields. Because different embedding models may produce embeddings of different dimensions and distribution types (dense vs. sparse), it is important to specify these two fields for Goodmem to store the embeddings properly.

The `jq` command at the end pretty-prints the JSON response for better readability. If your command above executes successfully, you should see a JSON response that looks like this:

```json
{
  "embedderId": "019b8229-11c7-76f2-b98b-89b29a7d50ce",
  "displayName": "OpenAI small",
  "description": "",
  "providerType": "OPENAI",
  "endpointUrl": "https://api.openai.com/v1",
  "apiPath": "/embeddings",
  "modelIdentifier": "text-embedding-3-small",
  "dimensionality": 1536,
  "distributionType": "DENSE",
  "maxSequenceLength": null,
  "supportedModalities": null,
  "labels": {},
  "version": "",
  "monitoringEndpoint": "",
  "ownerId": "b490757e-2d1e-4951-b111-9a6bd4435d1d",
  "createdAt": 1767415288264,
  "updatedAt": 1767415288264,
  "createdById": "b490757e-2d1e-4951-b111-9a6bd4435d1d",
  "updatedById": "b490757e-2d1e-4951-b111-9a6bd4435d1d"
}
```

The embedder is assigned a unique ID, `embedderId`, which we will use later. 

Optional fields that we did not specify in the registration command above include `description`, `labels`, `version`, `monitoringEndpoint`, and `supportedModalities`. They are returned as empty or null in the response. If you want, you can update these fields later. 

</details>


## Register the LLM

The command below registers the OpenAI `gpt-5-nano` LLM in Goodmem:

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
      "inlineSecret": "${OPENAI_API_KEY}"
    }
  }
}' | jq
```

<details>
<summary><strong>Expected response and analysis (click to expand)</strong></summary>

In this command, we set the LLM's display name as "My GPT-5-nano". Similar to the embedder registration command, we specify the `providerType`, `endpointUrl`, `modelIdentifier`, and `credentials` fields. Because LLM's output are always text, there is not other mandatory fields to specify for LLM registration.

If your command above executes successfully, you should see a JSON response that looks like this -- after being parsed by `jq`:

```json
{
  "llm": {
    "llmId": "019b8235-bca7-72da-a0d4-2bde72f68859",
    "displayName": "My GPT-5-nano",
    "description": null,
    "providerType": "OPENAI",
    "endpointUrl": "https://api.openai.com/v1",
    "apiPath": "/chat/completions",
    "modelIdentifier": "gpt-5-nano",
    "supportedModalities": [
      "TEXT"
    ],
    "labels": {},
    "version": null,
    "monitoringEndpoint": null,
    "capabilities": {
      "supportsChat": true,
      "supportsCompletion": true,
      "supportsFunctionCalling": true,
      "supportsSystemMessages": true,
      "supportsStreaming": true,
      "supportsSamplingParameters": false
    },
    "defaultSamplingParams": null,
    "maxContextLength": null,
    "clientConfig": null,
    "ownerId": "b490757e-2d1e-4951-b111-9a6bd4435d1d",
    "createdAt": 1767416118440,
    "updatedAt": 1767416118440,
    "createdById": "b490757e-2d1e-4951-b111-9a6bd4435d1d",
    "updatedById": "b490757e-2d1e-4951-b111-9a6bd4435d1d"
  },
  "statuses": [
    {
      "code": "LLM_CAPABILITY_INFERRED",
      "message": "LLM capability 'supports_chat' was automatically inferred as 'true' for model family 'gpt-5'"
    },
    {
      "code": "LLM_CAPABILITY_INFERRED",
      "message": "LLM capability 'supports_completion' was automatically inferred as 'true' for model family 'gpt-5'"
    },
    {
      "code": "LLM_CAPABILITY_INFERRED",
      "message": "LLM capability 'supports_function_calling' was automatically inferred as 'true' for model family 'gpt-5'"
    },
    {
      "code": "LLM_CAPABILITY_INFERRED",
      "message": "LLM capability 'supports_system_messages' was automatically inferred as 'true' for model family 'gpt-5'"
    },
    {
      "code": "LLM_CAPABILITY_INFERRED",
      "message": "LLM capability 'supports_streaming' was automatically inferred as 'true' for model family 'gpt-5'"
    },
    {
      "code": "LLM_CAPABILITY_INFERRED",
      "message": "LLM capability 'supports_sampling_parameters' was automatically inferred as 'false' for model family 'gpt-5'"
    }
  ]
}
```

Do not let the lengthy response intimidate you. It's long because Goodmem automatically infers many capabilities of the LLM based on its model family. The inference results are listed in the `statuses` field. 

Like the case for embedders, which returns the UUID of an embedder in the field `embedderId`, the LLM registration response returns the UUID of the LLM in the field `llmId`. We will use this ID later.

</details>

## Step 2: Add knowledge to the RAG agent

## Step 3: Spin the RAG agent 

## Step 4 (optional): Add a reranker
