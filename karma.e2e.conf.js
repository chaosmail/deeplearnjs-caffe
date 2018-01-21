module.exports = function(config) {
  config.set({
    frameworks: ["jasmine", "karma-typescript"],
    files: [
      { pattern: "src/**/*.ts" },
      { pattern: "test/**/*.ts" },
      { pattern: "test/**/*.jpg", watched: false, included: false },
      { pattern: "test/**/model/**/*", watched: false, included: false },
      { pattern: "test/**/activations/**/*", watched: false, included: false }
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
