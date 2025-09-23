# Release Notes

## Important Notes for This Release

### Deployment

Deploy all three applications:

- lil-lisa
- lillisa_server
- lil-lisa-web

### Post-Deployment Steps

- Rebuild the document store using one of the following commands:
  - `/rebuild_docs_contextual`
  - `/rebuild_docs_traditional`

### Testing
- After rebuilding is complete, run a query and verify that the citations are from the developer portal, not GitHub.
- **Note**: This may take some time.

## Standard Deployment Process

### Deployment

Deploy all three applications:

- lil-lisa
- lillisa_server
- lil-lisa-web

### Post-Deployment Steps

1. Verify that the new deployment is compatible with the existing document store.
2. Run one of the following Slack slash commands to rebuild the document store:
   - `/rebuild_docs_contextual`
   - `/rebuild_docs_traditional`
3. Note: Rebuilding the document store with the selected chunking strategy may take some time.
4. After rebuilding, test a query to confirm that the new chunking strategy is working as expected.