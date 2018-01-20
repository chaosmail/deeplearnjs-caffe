module.exports = function(config) {
  config.set({
    frameworks: ["jasmine", "karma-typescript"],
    files: [
      { pattern: "src/**/*.ts" },
      { pattern: "test/**/*.ts" },
      { pattern: "test/**/model/**/*", watched: false, included: false },
      { pattern: "test/**/activations/**/*", watched: false, included: false }
    ],
    preprocessors: { "**/*.ts": ["karma-typescript"]},
    karmaTypescriptConfig: {
      tsconfig: 'tsconfig.json'
    },
    reporters: ["progress", "karma-typescript"],
    browsers: ["Chrome"]
  });
};
