# Launching a RAG agent in 5 minutes with Goodmem

## Overview

RAG is an effective way to empower LLMs to think or decide based on external knowledge/information you provide even if such knowledge/information has not been "seen" by them before. 
For example, while it is impossible for a human supper enginer to memorize the manual of every product sold by a company,
a RAG-based customer service chatbot can answer any question about any product as long as it is provided with documentations of all products sold by the company.
On top of that, RAG is facinating in that it can always give the most up-to-date answer even if the product manuals are updated frequently.

Goodmem is framework for you to build, evaluate, and optimize RAG applications. 
While there are many RAG frameworks out there,
Goodmem is designed with enterprise-readiness in mind. [TODO: Elaborate why Goodmem is enterprise-ready.]
In this 5-minute tutorial, we will see why Goodmem is the best way to build enterprise-grade RAG applications.

In Goodmem, building a RAG application is as declaring the RAG components and leave the rest to Goodmem.
For example, the user does not have to deal with calling the components and passing data between them which they otherwise would have to do in other RAG frameworks.


## Before we start

1. Have an OpenAI API key (denoted as `$OPENAI_API_KEY`) ready. Optionally, please have a Voyager API key (denoted as `$VOYAGER_API_KEY`) ready for reranking. 
2. Install Goodmem: 
   ```bash
   curl -s "https://get.goodmem.ai" | bash
   ```
   [TODO: add flag for unattended install]
   The installation script will give you the path to Goodmem's REST API endpoint (denoted as `$GOODMEM_API_URL`) and your Goodmem API key (denoted as `$GOODMEM_API_KEY`). Be sure to write them down as we will use them in this tutorial.
3. Depending on your choice of interfacing with Goodmem, e.g., via CLI or Python, please install such interface. 
    For Python, please intall the Goodmem Python package:
    ```bash
    pip install goodmem-client
    ```
   

## Step 1: Register a RAG stack: an embedder and an LLM


In Goodmem, one RAG component can be used across many RAG knowledge bases or agents, while one RAG knowledge base/agent can employ multiple RAG components of the same type, e.g., two embedders, to balance different aspects of searching.

To support this design, every RAG component, once registered, is assigned a unique ID, UUID, easing the reference and management of RAG components in an enterprise setting.    
In contrast, many existing RAG frameworks do not enforce this mechanism and duplicated instances of RAG components of same settings  cause management overhead and confusion.

Also to support this design, each component is a microservice that can be and must be access via gRPC/REST calls. Therefore, your RAG pipeline is not tied to the programming language you use to build it. 

At its bare minimal, a RAG application consists of two components: an embedder and a large language model (LLM).
In this tutorial, we will use OpenAI's `text-embedding-3-small` embedder and OpenAI's `gpt-5-nano` LLM.
We will register these two components via Goodmem's REST API. 

First, let's declare two Goodmem-related environment variables (if you have not done so):

```bash
export GOODMEM_BASE_URL="your_goodmem_api_url"  # e.g., http://localhost:8080/v1. Must end with /v1
export GOODMEM_API_KEY="your_goodmem_api_key"    # replace with your Goodmem API KEY
```

Then, let's registered the embedder:

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
            "kind": "CREDENTIAL_KIND_API_KEY",
            "apiKey": {
            "inlineSecret": "${OPENAI_API_KEY}"
            }
        }
    }' | jq
```

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

As the last task of Step 1, let's register the LLM:   


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