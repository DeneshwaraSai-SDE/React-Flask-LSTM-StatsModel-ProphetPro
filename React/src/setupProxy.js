const { createProxyMiddleware } = require("http-proxy-middleware");

/*
This acts a proxy between the react application and the flask microservice
Everytime there is a request to /api, the setupProxy prepends the flask
microservice url mentioned in line 14
*/
module.exports = function (app) {
  app.use(
    "/api",
    createProxyMiddleware({
      // update the flask Google Cloud url
      // target: "https://assignment-5-flask-4-700937886554.us-central1.run.app",
      target: "https://assignment-5-flask-5-700937886554.us-central1.run.app",
      changeOrigin: true,
    })
  );
};