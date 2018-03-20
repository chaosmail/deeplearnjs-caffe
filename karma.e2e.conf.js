module.exports = function(config) {
  config.set({
    frameworks: ["jasmine", "karma-typescript"],
    files: [
      { pattern: "src/**/*.ts" },
      { pattern: "test/e2e/**/*.ts" },
      { pattern: "test/e2e/assets/**/*", watched: false, included: false },
      { pattern: "test/e2e/**/model/**/*", watched: false, included: false },
      { pattern: "test/e2e/**/activations/**/*", watched: false, included: false }
    ],
    preprocessors: { "**/*.ts": ["karma-typescript"]},
    karmaTypescriptConfig: {
      tsconfig: 'tsconfig.json'
    },
    reporters: ["progress", "karma-typescript"],
    browsers: ["Chrome"],

    // Wait maximal 5 min for a single test.
    // this is needed for CPU pass through a model
    browserNoActivityTimeout: 1000*60*5
  });
};
