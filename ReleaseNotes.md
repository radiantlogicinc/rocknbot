# Release Notes

## Important Notes for This Release

### Changed version

- lil-lisa - from 2.2.4 to 2.3.0
- lillisa_server - from 2.5.2 to 2.6.0
- lil-lisa-web - from 2.3.4 to 2.4.0

- LDB_TAG Changed from 2.5.3 to 2.6.0 (Is this necessary?)

### Testing
- After rebuilding is complete (Roughly one hour), run a query and verify that the citations are from the developer portal, not GitHub.
- **Note**: There is no notification when the rebuild is complete.

## Standard Deployment Process

### Deployment

Deploy only those applications where the version has changed:

- lil-lisa
- lillisa_server
- lil-lisa-web

### Post-Deployment Steps

- Rebuild the document store using the following commands on slack lil-lisa admin channel:
  - `/rebuild_docs_contextual`

### Testing
- After rebuilding is complete (Roughly one hour), run a query and verify the response.
- **Note**: There is no notification when the rebuild is complete.