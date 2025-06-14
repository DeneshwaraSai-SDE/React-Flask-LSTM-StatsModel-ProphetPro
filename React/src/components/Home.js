/*
Goal of React:
  1. React will retrieve GitHub created and closed issues for a given repository and will display the bar-charts 
     of same using high-charts        
  2. It will also display the images of the forecasted data for the given GitHub repository and images are being retrieved from 
     Google Cloud storage
  3. React will make a fetch api call to flask microservice.
*/

// Import required libraries
import * as React from "react";
import { useState } from "react";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import AppBar from "@mui/material/AppBar";
import CssBaseline from "@mui/material/CssBaseline";
import Toolbar from "@mui/material/Toolbar";
import List from "@mui/material/List";
import Typography from "@mui/material/Typography";
import Divider from "@mui/material/Divider";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
// Import custom components
import BarCharts from "./BarCharts";
import Loader from "./Loader";
import { Card, ListItemButton } from "@mui/material";
import axios from "axios";

const drawerWidth = 240;
// List of GitHub repositories
const repositories = [
  {
    key: "langchain-ai/langgraph",
    value: "langgraph",
  },
  {
    key: "ollama/ollama",
    value: "ollama",
  },
  {
    key: "meta-llama/llama3",
    value: "llama3",
  },
  {
    key: "langchain-ai/langchain",
    value: "langchain",
  },
  {
    key: "microsoft/autogen",
    value: "autogen",
  },
  { key: "openai/openai-cookbook", value: "openai-cookbook" },
  { key: "elastic/elasticsearch", value: "elasticsearch" },
  { key: "milvus-io/pymilvus", value: "pymilvus" },
  {
    key: "angular/angular",
    value: "Angular",
  },
  {
    key: "angular/angular-cli",
    value: "Angular-cli",
  },
  {
    key: "angular/material",
    value: "Angular Material",
  },
  {
    key: "d3/d3",
    value: "D3",
  },
  { key: "ollama/ollama", value: "ollama" },
];

export default function Home() {
  /*
  The useState is a react hook which is special function that takes the initial 
  state as an argument and returns an array of two entries. 
  */
  /*
  setLoading is a function that sets loading to true when we trigger flask microservice
  If loading is true, we render a loader else render the Bar charts
  */
  const [loading, setLoading] = useState(true);
  /* 
  setRepository is a function that will update the user's selected repository such as Angular,
  Angular-cli, Material Design, and D3
  The repository "key" will be sent to flask microservice in a request body
  */
  const [repository, setRepository] = useState({
    key: "langchain-ai/langgraph",
    value: "langgraph",
  });
  /*
  
  The first element is the initial state (i.e. githubRepoData) and the second one is a function 
  (i.e. setGithubData) which is used for updating the state.

  so, setGitHub data is a function that takes the response from the flask microservice 
  and updates the value of gitHubrepo data.
  */
  const [githubRepoData, setGithubData] = useState({});
  // Updates the repository to newly selected repository
  const eventHandler = (repo) => {
    setRepository(repo);
  };

  /* 
  Fetch the data from flask microservice on Component load and on update of new repository.
  Everytime there is a change in a repository, useEffect will get triggered, useEffect inturn will trigger 
  the flask microservice 
  */
  React.useEffect(() => {
    // set loading to true to display loader
    setLoading(true);
    const requestOptions = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      // Append the repository key to request body
      body: JSON.stringify({ repository: repository.key }),
    };

    /*
    Fetching the GitHub details from flask microservice
    The route "/api/github" is served by Flask/App.py in the line 53
    @app.route('/api/github', methods=['POST'])
    Which is routed by setupProxy.js to the
    microservice target: "your_flask_gcloud_url"
    */

 
    try {
      axios
        .post(
          "https://assignment-5-flask-5-700937886554.us-central1.run.app/api/github",
          { repository: repository.key },
          { requestOptions },
          {
            headers: {
              "Content-Type": "application/json",
            },
            withCredentials: false, // set true only if sending cookies
          }
        )
        .then((res) => {
          console.log(res);
          setGithubData(res.data);
          setLoading(false);
        })
        .error((Err) => {
          setLoading(false);
          setGithubData({});
        });
    } catch (error) {
      console.error("Error fetching data:", error);
      setLoading(false);
      setGithubData({});
    }

    // fetch("/api/githubs", requestOptions)
    //   .then((res) => res.json())
    //   .then(
    //     // On successful response from flask microservice
    //     (result) => {
    //       // On success set loading to false to display the contents of the resonse
    //       setLoading(false);
    //       // Set state on successfull response from the API
    //       setGithubData(result);
    //     },
    //     // On failure from flask microservice
    //     (error) => {
    //       // Set state on failure response from the API
    //       console.log(error);
    //       // On failure set loading to false to display the error message
    //       setLoading(false);
    //       setGithubData([]);
    //     }
    //   );
  }, [repository]);

  return (
    <Box sx={{ display: "flex" }}>
      <CssBaseline />
      {/* Application Header */}
      <AppBar
        position="fixed"
        sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}
      >
        <Toolbar>
          <Typography variant="h6" noWrap component="div">
            Timeseries Forecasting
          </Typography>
        </Toolbar>
      </AppBar>
      {/* Left drawer of the application */}
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: {
            width: drawerWidth,
            boxSizing: "border-box",
          },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: "auto" }}>
          <List>
            {/* Iterate through the repositories list */}
            {repositories.map((repo) => (
              <ListItem
                button
                key={repo.key}
                onClick={() => eventHandler(repo)}
                disabled={loading && repo.value !== repository.value}
              >
                <ListItemButton selected={repo.value === repository.value}>
                  <ListItemText primary={repo.value} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Toolbar />
        {/* Render loader component if loading is true else render charts and images */}
        {loading ? (
          <Loader />
        ) : (
          <div>
            {/* Render barchart component for a monthly created issues for a selected repositories*/}
            <BarCharts
              title={`Monthly Created Issues for ${repository.value} in last 1 year`}
              data={githubRepoData?.created}
            />
            {/* Render barchart component for a monthly created issues for a selected repositories*/}
            <BarCharts
              title={`Monthly Closed Issues for ${repository.value} in last 1 year`}
              data={githubRepoData?.closed}
            />
            <Divider
              sx={{ borderBlockWidth: "3px", borderBlockColor: "#FFA500" }}
            />
            {/* Rendering Timeseries Forecasting of Created Issues using Tensorflow and
                Keras LSTM */}
            {/* <div>
              <Typography variant="h5" component="div" gutterBottom>
                Timeseries Forecasting of Created Issues using Tensorflow and
                Keras LSTM based on past month
              </Typography>

              <div>
                <Typography component="h4">
                  Model Loss for Created Issues
                </Typography>
                <img
                  src={githubRepoData?.createdAtImageUrls?.model_loss_image_url}
                  alt={"Model Loss for Created Issues"}
                  loading={"lazy"}
                />
              </div>
              <div>
                <Typography component="h4">
                  LSTM Generated Data for Created Issues
                </Typography>
                <img
                  src={
                    githubRepoData?.createdAtImageUrls?.lstm_generated_image_url
                  }
                  alt={"LSTM Generated Data for Created Issues"}
                  loading={"lazy"}
                />
              </div>
              <div>
                <Typography component="h4">
                  All Issues Data for Created Issues
                </Typography>
                <img
                  src={
                    githubRepoData?.createdAtImageUrls?.all_issues_data_image
                  }
                  alt={"All Issues Data for Created Issues"}
                  loading={"lazy"}
                />
              </div>
            </div>
      
            <div>
              <Divider
                sx={{ borderBlockWidth: "3px", borderBlockColor: "#FFA500" }}
              />
              <Typography variant="h5" component="div" gutterBottom>
                Timeseries Forecasting of Closed Issues using Tensorflow and
                Keras LSTM based on past month
              </Typography>

              <div>
                <Typography component="h4">
                  Model Loss for Closed Issues
                </Typography>
                <img
                  src={githubRepoData?.closedAtImageUrls?.model_loss_image_url}
                  alt={"Model Loss for Closed Issues"}
                  loading={"lazy"}
                />
              </div>
              <div>
                <Typography component="h4">
                  LSTM Generated Data for Closed Issues
                </Typography>
                <img
                  src={
                    githubRepoData?.closedAtImageUrls?.lstm_generated_image_url
                  }
                  alt={"LSTM Generated Data for Closed Issues"}
                  loading={"lazy"}
                />
              </div>
              <div>
                <Typography component="h4">
                  All Issues Data for Closed Issues
                </Typography>
                <img
                  src={githubRepoData?.closedAtImageUrls?.all_issues_data_image}
                  alt={"All Issues Data for Closed Issues"}
                  loading={"lazy"}
                />
              </div>
            </div> */}

            <br></br>

            <br></br>
            <div>
              FORT COUNTS AND STARS COUNT for {repository.value}
              <div>
                <Card style={{ padding: 8, paddingTop: 18, marginBottom: 5 }}>
                  <Typography variant="h5" component="div" gutterBottom>
                    FORK COUNT: {githubRepoData?.forkCount}
                  </Typography>
                </Card>

                <Card style={{ padding: 8, paddingTop: 18 }}>
                  <Typography variant="h5" component="div" gutterBottom>
                    STARS COUNT: {githubRepoData?.starCount}
                  </Typography>
                </Card>
              </div>
            </div>

            <br></br>

            <br></br>

            <div>
              <Typography variant="h5" component="div" gutterBottom>
                <b>Issues Created At for {repository.value}</b>
              </Typography>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of Issues
                  Created At per month using Tensorflow and Keras LSTM
                </Typography>
                <img
                  src={
                    githubRepoData?.issues_created_at_all_models_response
                      ?.forecast_lstm_issues_created_at_url
                  }
                  alt={"forecast_lstm_issues_created_at_url"}
                  loading={"lazy"}
                />
              </div>
              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of Issues
                  Created At per month using statsmodel
                </Typography>
                <img
                  src={
                    githubRepoData?.issues_created_at_all_models_response
                      ?.forecast_stats_issues_created_at_url
                  }
                  alt={"forecast_stats_issues_created_at_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of Issues
                  Created At per month using Prophet Model
                </Typography>
                <img
                  src={
                    githubRepoData?.issues_created_at_all_models_response
                      ?.fbprophet_forecast_issues_created_at_components_url
                  }
                  alt={"forecast_stats_issues_created_at_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of Issues
                  Created At per month using Prophet Model
                </Typography>
                <img
                  src={
                    githubRepoData?.issues_created_at_all_models_response
                      ?.fbprophet_forecast_issues_created_at_url
                  }
                  alt={"forecast_stats_issues_created_at_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Output for day of the week for {repository.value}
                </Typography>
                <h5>
                  {
                    githubRepoData?.issues_created_at_all_models_response
                      ?.week_of_the_day
                  }
                </h5>
              </div>
            </div>

            <br></br>

            <br></br>

            <div>
              <Typography variant="h5" component="div" gutterBottom>
                Issues Closed At
              </Typography>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of Issues
                  Closed At per month using Tensorflow and Keras LSTM for{" "}
                  {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.issues_closed_at_all_models_response
                      ?.forecast_lstm_issues_closed_at_url
                  }
                  alt={"forecast_lstm_issues_closed_at_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of Issues
                  Closed At per month using statsmodel for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.issues_closed_at_all_models_response
                      ?.forecast_stats_issues_closed_at_url
                  }
                  alt={"forecast_stats_issues_closed_at_url"}
                  loading={"lazy"}
                />
              </div>


              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of Issues
                  Closed At per month using Prophet Model
                </Typography>
                <img
                  src={
                    githubRepoData?.issues_closed_at_all_models_response
                      ?.fbprophet_forecast_issues_closed_at_components_url
                  }
                  alt={"forecast_stats_issues_created_at_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of Issues
                  Closed At per month using Prophet Model
                </Typography>
                <img
                  src={
                    githubRepoData?.issues_closed_at_all_models_response
                      ?.fbprophet_forecast_issues_closed_at_url 
                  }
                  alt={"forecast_stats_issues_created_at_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Output for day of the week and month of the year for{" "}
                  {repository.value}
                </Typography>
                <h5>
                  {
                    githubRepoData?.issues_closed_at_all_models_response
                      ?.week_of_the_day
                  }
                </h5>
                <h5>
                  {
                    githubRepoData?.issues_closed_at_all_models_response
                      ?.month_of_the_year
                  }
                </h5>
              </div>
            </div>

            <br></br>

            <br></br>

            <div>
              <Typography variant="h5" component="div" gutterBottom>
                USING PULLS DATA
              </Typography>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of PULLS
                  data using Tensorflow and Keras LSTM for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.pulls_all_models_response
                      ?.forecast_lstm_pulls_url
                  }
                  alt={"forecast_lstm_pulls_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of PULLS
                  data using statsmodel for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.pulls_all_models_response
                      ?.forecast_stats_pulls_url
                  }
                  alt={"forecast_stats_pulls_url"}
                  loading={"lazy"}
                />
              </div>

                  
              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of PULLS
                  data using prophet model for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.pulls_all_models_response
                      ?.fbprophet_forecast_pulls_components_url
                  }
                  alt={"forecast_stats_pulls_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of PULLS
                  data using prophet model for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.pulls_all_models_response
                      ?.fbprophet_forecast_pulls_url
                  }
                  alt={"forecast_stats_pulls_url"}
                  loading={"lazy"}
                />
              </div>

            </div>

            <br></br>

            <br></br>

            <div>
              <Typography variant="h5" component="div" gutterBottom>
                USING COMMITS DATA
              </Typography>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of COMMITS
                  data using Tensorflow and Keras LSTM for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.commits_all_models_response
                      ?.forecast_lstm_commits_url
                  }
                  alt={"forecast_lstm_commits_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of COMMITS
                  data using statsmodel for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.commits_all_models_response
                      ?.forecast_stats_commits_url
                  }
                  alt={"forecast_stats_commits_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of COMMITS
                  data using Prophet for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.commits_all_models_response
                      ?.fbprophet_forecast_commits_components_url
                  }
                  alt={"forecast_stats_commits_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of COMMITS
                  data using Prophet for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.commits_all_models_response
                      ?.fbprophet_forecast_commits_url
                  }
                  alt={"fbprophet_forecast_commits_url"}
                  loading={"lazy"}
                />
              </div>
            </div>

            <br></br>

            <br></br>

            <div>
              <Typography variant="h5" component="div" gutterBottom>
                USING BRANCHES DATA
              </Typography>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of BRANCHES
                  data using Tensorflow and Keras LSTM for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.branch_all_models_response
                      ?.forecast_lstm_branch_url
                  }
                  alt={"forecast_lstm_branch_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of BRANCHES
                  data using statsmodel for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.branch_all_models_response
                      ?.forecast_stats_branch_url
                  }
                  alt={"forecast_stats_branch_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of BRANCHES
                  data using prophet for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.branch_all_models_response
                      ?.fbprophet_forecast_branch_components_url
                  }
                  alt={"forecast_stats_branch_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of BRANCHES
                  data using prophet for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.branch_all_models_response
                      ?.fbprophet_forecast_branch_url
                  }
                  alt={"forecast_stats_branch_url"}
                  loading={"lazy"}
                />
              </div>
            </div>

            <br></br>

            <br></br>

            <div>
              <Typography variant="h5" component="div" gutterBottom>
                USING CONTRUBUTORS DATA
              </Typography>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of
                  CONTRUBUTORS data using Tensorflow and Keras LSTM for{" "}
                  {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.contributors_all_models_response
                      ?.forecast_lstm_contributors_url
                  }
                  alt={"forecast_lstm_contributors_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of BRANCHES
                  data using statsmodel for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.contributors_all_models_response
                      ?.forecast_stats_contributors_url
                  }
                  alt={"forecast_stats_contributors_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of BRANCHES
                  data using prophet for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.contributors_all_models_response
                      ?.fbprophet_forecast_contributors_components_url
                  }
                  alt={"forecast_stats_contributors_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of BRANCHES
                  data using prophet for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.contributors_all_models_response
                      ?.fbprophet_forecast_contributors_url
                  }
                  alt={"forecast_stats_contributors_url"}
                  loading={"lazy"}
                />
              </div>
            </div>

            <br></br>

            <br></br>

            <div>
              <Typography variant="h5" component="div" gutterBottom>
                USING RELEASES DATA
              </Typography>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of RELEASES
                  data using Tensorflow and Keras LSTM for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.releases_all_models_response
                      ?.forecast_lstm_releases_url
                  }
                  alt={"forecast_lstm_contributors_url"}
                  loading={"lazy"}
                />
              </div>

              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of RELEASES
                  data using statsmodel for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.releases_all_models_response
                      ?.forecast_stats_releases_url
                  }
                  alt={"forecast_stats_contributors_url"}
                  loading={"lazy"}
                />
              </div>
              
              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of RELEASES
                  data using prophet for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.releases_all_models_response
                      ?.fbprophet_forecast_releases_components_url

                  }
                  alt={"forecast_stats_contributors_url"}
                  loading={"lazy"}
                />
              </div>
              
              <div>
                <Typography variant="h5" component="div">
                  Graph for historical data and Time Series Forcast of RELEASES
                  data using prophet for {repository.value}
                </Typography>
                <img
                  src={
                    githubRepoData?.releases_all_models_response
                      ?.fbprophet_forecast_releases_url
                  }
                  alt={"forecast_stats_contributors_url"}
                  loading={"lazy"}
                />
              </div>
            </div>

            <br></br>

            <br></br>
          </div>
        )}
      </Box>
    </Box>
  );
}
