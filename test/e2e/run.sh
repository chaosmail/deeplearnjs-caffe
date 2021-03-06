#!/bin/bash

# Run caffe inference
sh "$(dirname $0)/squeezenet/run.sh" --fetch
sh "$(dirname $0)/googlenet/run.sh" --fetch

# Run Karma test runner
node_modules/.bin/karma start karma.e2e.conf.js $@
