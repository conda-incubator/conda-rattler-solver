# How to cut a new release

Given an example release with version `0.1.0`:

1. Make sure `main` is up-to-date and create a new branch: `git checkout -b prepare-0.1.0`.
2. In Github Releases, draft a new release with tag `0.1.0` and use the "Generate release notes" button. Copy the generated text.
3. Paste the text from step 2 in a new section in `CHANGELOG.md`. Make the necessary edits for clarity (e.g. fix typos, categorize as needed, etc).
4. Commit and push the branch, opening a new PR in the process.
5. Once reviewed and merged, use the edited text in `CHANGELOG.md` as the release notes for the Github Releases publication.
