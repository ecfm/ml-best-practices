# Shared Learnings
At the start of each session, pull shared learnings: `cd ~/ml-best-practices && git pull --quiet`
For detailed ML best practices, read ~/ml-best-practices/README.md before starting training or inference work.
Use `/sync-learnings` to commit and push any new learnings at end of session.

# General Rules
always use uv to manage dependency and run python
don't use try catch to silent errors, always let it fail-fast. only apply try catch when there is no alternative and warn the user
don't use .get with default values. if key missing just let it throw exception. warn user if this has to be used
for ML task, save detailed logs, inputs, intermediate results, output, metrics, and model specs with easy to find structure for later debugging
after update or fix, always run a small but complete test by yourself and debug it until it succeeds, prior to running a long job that is newly implemented or updated
after running experiments, carefully review logs, inputs, intermediate results, output, metrics, and model specs to check whether all steps follow expectation, identify potential issues and improvement directions.