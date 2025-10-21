# Release Notes

## Important Notes for This Release

**Current Production Versions:**
- lil-lisa: 2.2.4
- lillisa_server: 2.5.2
- lil-lisa-web: 2.3.4

**Changed Production Version:**
- lil-lisa: 2.3.1 (includes changes from undeployed 2.3.0 + new features)
- lillisa_server: 2.6.1 (includes changes from undeployed 2.6.0 + new features)
- lil-lisa-web: 2.4.1 (includes changes from undeployed 2.4.0 + new features)

**Note:** This release includes all changes from the previously submitted but undeployed release.

- LDB_TAG Changed from 2.5.3 to 2.6.0 (Is this necessary?)

## Golden QA Pairs Repository Update
Create a new `file ido_qa_pairs.md` in the golden QA pairs repository with the following sample data:
``` 
# Question/Answer Pair 1

Question: What is IDO?

Answer: IDO (Identity Data Orchestration) is RadiantLogic's solution for managing identity data across multiple systems.
```


## Pre-Deployment Action
Delete the existing LanceDB datastore before deploying this release. A fresh datastore with all documents will be automatically created and populated upon deployment.

**Note**: This process takes approximately 2 hours to complete. There is no notification when the rebuild is complete.

## New environment variables
### Server Configuration (lillisa_server.env)
1. Add the following environment variables:
    - `DOCUMENTATION_IDO_VERSIONS="v1.0"`
    - `IDO_PRODUCT_VERSIONS="dev/v1.0, v1.0"`

#### Slack Configuration (lil-lisa.env)
1. Create two new Slack channels:
    - `lil-ido`
    - `lil-ido-admin`
2. Add the following environment variables:
    - `CHANNEL_ID_IDO` - Set this to the channel ID of `lil-ido`
    - `ADMIN_CHANNEL_ID_IDO` - Set this to the channel ID of `lil-ido-admin`
    - `EXPERT_USER_ID_IDO` - Set this to the appropriate expert user ID

## Standard Deployment Process

### Deployment

Deploy only those applications where the version has changed:

- lil-lisa
- lillisa_server
- lil-lisa-web
 
### Testing (Wait approximately 2 hours after deployment)

1. Run a query and verify that citations are sourced from the developer portal, not GitHub
2. Test the new IDO product by running a query in the `lil-ido` channel
    - Example query: "In Graph Pipelines Configuration can you explain me Structure of a configuration."
    - Expected output should include: Source Objects, Vertices, Edges, Functions
