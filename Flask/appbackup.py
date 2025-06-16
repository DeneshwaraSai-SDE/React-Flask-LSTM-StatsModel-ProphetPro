'''
Goal of Flask Microservice:
1. Flask will take the repository_name such as angular, angular-cli, material-design, D3 from the body of the api sent from React app and 
   will utilize the GitHub API to fetch the created and closed issues. Additionally, it will also fetch the author_name and other 
   information for the created and closed issues.
2. It will use group_by to group the data (created and closed issues) by month and will return the grouped data to client (i.e. React app).
3. It will then use the data obtained from the GitHub API (i.e Repository information from GitHub) and pass it as a input request in the 
   POST body to LSTM microservice to predict and forecast the data.
4. The response obtained from LSTM microservice is also return back to client (i.e. React app).

Use Python/GitHub API to retrieve Issues/Repos information of the past 1 year for the following repositories:
- https: // github.com/angular/angular
- https: // github.com/angular/material
- https: // github.com/angular/angular-cli
- https: // github.com/d3/d3
'''
# Import all the required packages 
import os
from flask import Flask, jsonify, request, make_response, Response
from flask_cors import CORS
import json
import dateutil.relativedelta
from dateutil import *
from datetime import date
import pandas as pd
import requests
from datetime import datetime as datetime,timedelta, timezone
# Initilize flask app
app = Flask(__name__)
# Handles CORS (cross-origin resource sharing)
CORS(app)

async def async_get_request(session, url, headers=None, params=None):
    async with session.get(url, headers=headers, params=params) as response:
        return await response.json()

async def async_post_request(session, url, json_data, headers=None):
    async with session.post(url, json=json_data, headers=headers) as response:
        return response.status, await response.json()

# Add response headers to accept all types of  requests
def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

# Modify response headers when returning to the origin
def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

'''
API route path is  "/api/forecast"
This API will accept only POST request
'''
@app.route('/api/github', methods=['POST'])
def github():
    print("STARTED GITHUB API PROCESSING")
    body = request.get_json()
    repo_name = body['repository']

    GITHUB_TOKEN = ''
    token = os.environ.get('GITHUB_TOKEN', GITHUB_TOKEN)
    
    # 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
    # NEW 

    GITHUB_URL = f"https://api.github.com/"
    headers = {
        "Authorization": f'token {token}'
    }
    params = {
        "state": "open"
    }
    repository_url = GITHUB_URL + "repos/" + repo_name
    repository = requests.get(repository_url, headers=headers)
    repository = repository.json()

    today = date.today()

    issues_reponse = []
    # Iterating to get issues for every month for the past 12 months
    for i in range(12):
        last_month = today + dateutil.relativedelta.relativedelta(months=-1)
        types = 'type:issue'
        repo = 'repo:' + repo_name
        ranges = 'created:' + str(last_month) + '..' + str(today)
        # By default GitHub API returns only 30 results per page
        # The maximum number of results per page is 100
        # For more info, visit https://docs.github.com/en/rest/reference/repos 
        per_page = 'per_page=100'
        # Search query will create a query to fetch data for a given repository in a given time range
        search_query = types + ' ' + repo + ' ' + ranges

        # Append the search query to the GitHub API URL 
        query_url = GITHUB_URL + "search/issues?q=" + search_query + "&" + per_page
        # requsets.get will fetch requested query_url from the GitHub API
        search_issues = requests.get(query_url, headers=headers, params=params)
        # Convert the data obtained from GitHub API to JSON format
        search_issues = search_issues.json()
        issues_items = []
        try:
            # Extract "items" from search issues
            issues_items = search_issues.get("items")
        except KeyError:
            error = {"error": "Data Not Available"}
            resp = Response(json.dumps(error), mimetype='application/json')
            resp.status_code = 500
            return resp
        if issues_items is None:
            continue
        for issue in issues_items:
            label_name = []
            data = {}
            current_issue = issue
            # Get issue number
            data['issue_number'] = current_issue["number"]
            # Get created date of issue
            data['created_at'] = current_issue["created_at"][0:10]
            if current_issue["closed_at"] == None:
                data['closed_at'] = current_issue["closed_at"]
            else:
                # Get closed date of issue
                data['closed_at'] = current_issue["closed_at"][0:10]
            for label in current_issue["labels"]:
                # Get label name of issue
                label_name.append(label["name"])
            data['labels'] = label_name
            # It gives state of issue like closed or open
            data['State'] = current_issue["state"]
            # Get Author of issue
            data['Author'] = current_issue["user"]["login"]
            issues_reponse.append(data)

        today = last_month
        print(f"END OF ISSUES FOR PAGE : {i}")

    df = pd.DataFrame(issues_reponse)

    # Daily Created Issues
    df_created_at = df.groupby(['created_at'], as_index=False).count()
    dataFrameCreated = df_created_at[['created_at', 'issue_number']]
    dataFrameCreated.columns = ['date', 'count']

    '''
    Monthly Created Issues
    Format the data by grouping the data by month
    ''' 
    created_at = df['created_at']
    month_issue_created = pd.to_datetime(
        pd.Series(created_at), format='%Y-%m-%d')
    month_issue_created.index = month_issue_created.dt.to_period('m')
    month_issue_created = month_issue_created.groupby(level=0).size()
    month_issue_created = month_issue_created.reindex(pd.period_range(
        month_issue_created.index.min(), month_issue_created.index.max(), freq='m'), fill_value=0)
    month_issue_created_dict = month_issue_created.to_dict()
    created_at_issues = []
    for key in month_issue_created_dict.keys():
        array = [str(key), month_issue_created_dict[key]]
        created_at_issues.append(array)

    '''
    Monthly Closed Issues
    Format the data by grouping the data by month
    ''' 
    
    closed_at = df['closed_at'].sort_values(ascending=True)
    month_issue_closed = pd.to_datetime(
        pd.Series(closed_at), format='%Y-%m-%d')
    month_issue_closed.index = month_issue_closed.dt.to_period('m')
    month_issue_closed = month_issue_closed.groupby(level=0).size()
    month_issue_closed = month_issue_closed.reindex(pd.period_range(
        month_issue_closed.index.min(), month_issue_closed.index.max(), freq='m'), fill_value=0)
    month_issue_closed_dict = month_issue_closed.to_dict()
    closed_at_issues = []
    for key in month_issue_closed_dict.keys():
        array = [str(key), month_issue_closed_dict[key]]
        closed_at_issues.append(array)


    print(created_at_issues)
    print(closed_at_issues)
 
    print("DONE FETCHING ISSUES:\n")

    #CODE FOR PULL REQUESTS
    end_date = datetime.now()
    start_date = end_date - timedelta(days = 365 * 2)
    pull_response = []
    n = 0
    for i in range(6):
        per_page = 'per_page=100'
        page = 'page='
        search_query = repo_name + '/pulls?state=all' + "&" + per_page+ "&" + page + f'{i}'
        query_url = GITHUB_URL + "repos/" + search_query
        print(query_url)
        pull_requests = requests.get(query_url, headers=headers, params=params)
        pull_items = pull_requests.json()

        if pull_items is None:
            continue
        for pull_req in pull_items:
            label_name = []
            data = {}
            current_pull_req = pull_req
            created_at_date = datetime.strptime(current_pull_req["created_at"][0:10], "%Y-%m-%d")
            max_date = datetime.now() - timedelta(days=365 * 2)

            if created_at_date > max_date:
                data['pull_req_number'] = current_pull_req["number"]
                data['created_at'] = current_pull_req["created_at"][0:10]
                if current_pull_req["closed_at"] == None:
                    data['closed_at'] = current_pull_req["closed_at"]
                else:
                    data['closed_at'] = current_pull_req["closed_at"][0:10]
                for label in current_pull_req["labels"]:
                    label_name.append(label["name"])
                data['labels'] = label_name
                data['State'] = current_pull_req["state"]
                data['Author'] = current_pull_req["user"]["login"]
                pull_response.append(data)
        print("END OF Part - ", i)
    print("pulls:\n")
 

    ## BRANCH RESPONSE
    branch_response = []

    n = 0
    for i in range(12):
        per_page = 'per_page=100'
        page = 'page='
        search_query = repo_name + '/branches?state=all' + "&" + per_page+ "&" + page + f'{i}'
        query_url = GITHUB_URL + "repos/" + search_query
        branch_requests = requests.get(query_url, headers=headers, params=params)
        branch_items = branch_requests.json()
    
        if branch_items is None:
            continue
        for branch_req in branch_items:
            current_branch_req = branch_req 
            # print(current_branch_req)
            if current_branch_req['commit']['sha'] is None: 
                print("NO data found")

            if current_branch_req['commit']['sha'] is not None: 
                branch_commit_url = current_branch_req['commit']['url'] 
                # f"https://api.github.com/repos/{repo_name}/commits/{current_branch_req['commit']['sha']}"

                response = requests.get(branch_commit_url, headers = headers)
                commitedDetails = response.json()
                
                commitDate = datetime.strptime(commitedDetails['commit']['author']['date'], "%Y-%m-%dT%H:%M:%SZ") 
                print(commitDate)
                max_date = datetime.now() - timedelta(days=730)
                if commitDate > max_date:
                    branch_response.append({ 'commit_date': commitedDetails['commit']['author']['date'] })
        print(f"END OF BRNACHES FOR PAGE : {i}")
    print("branchs:\n")


    ## CONTRIBUTORS RESPONSE
    contributors_response = []

    end_date = datetime.now()
    start_date = end_date - timedelta(days = 365 * 2)

    n = 0
    for i in range(6):
        per_page = 'per_page=100'
        page = 'page='
        search_query = repo_name + '/contributors?state=all' + "&" + per_page+ "&" + page + f'{i}'
        query_url = GITHUB_URL + "repos/" + search_query
        contributors_requests = requests.get(query_url, headers=headers, params=params)
        contributors_items = contributors_requests.json()
    
        if contributors_items is None:
            continue
        for contributors_req in contributors_items:
            label_name = []
            data = {}
            login = contributors_req['login']

            userURL = f"https://api.github.com/users/{login}/events/public"
            eventRes = requests.get(userURL, headers=headers, params={"per_page": 100})
    
            if eventRes.status_code == 200:
                events = eventRes.json()
                for event in events:
                    createdEventDate = event.get("created_at")
                    if createdEventDate:
                        event_datetime = datetime.fromisoformat(createdEventDate[:-1])
                        if start_date <= event_datetime <= end_date:
                            contributors_response.append({
                                "contributor": login,
                                "event_date": createdEventDate,
                            })
                            break
        print(f"END OF CONTRIBUTIONS FOR PAGE : {i}")                    
        
    print("contributors:\n")



    ## RELEASE RESPONSE
    releases_response = []

    end_date = datetime.now()
    start_date = end_date - timedelta(days = 365 * 2)

    n = 0
    for i in range(12):
        per_page = 'per_page=100'
        page = 'page='
        search_query =repo_name + '/releases?state=all' + "&" + per_page+ "&" + page + f'{i}'
        query_url = GITHUB_URL + "repos/" + search_query
        releases_requests = requests.get(query_url, headers=headers, params=params)
    
        releases_items = releases_requests.json()

        if releases_items is None or len(releases_items) == 0:
            break
        for release in releases_items:
            start_date1 = start_date.replace(tzinfo=timezone.utc)
            end_date1 = end_date.replace(tzinfo=timezone.utc)
            published_at = datetime.fromisoformat(release['published_at'].replace("Z", "+00:00"))
            if start_date1 <= published_at <= end_date1:
                releases_response.append({ 
                    "published_at": release['published_at'], 
                })
        print(f"END OF RELEASES FOR PAGE : {i}")
    print("releases:\n")


    ## GET commits DATA
    commits_response = []

    end_date = datetime.now()
    start_date = end_date - timedelta(days = 365 * 2)

    n = 0
    for i in range(12):
        per_page = 'per_page=100'
        page = 'page='
        search_query = repo_name + '/commits?state=all' + "&" + per_page+ "&" + page + f'{i}'
        query_url = GITHUB_URL + "repos/" + search_query
        commits_requests = requests.get(query_url, headers=headers, params=params)
    
        commits_items = commits_requests.json()

        if commits_items is None or len(commits_items) == 0:
            break
        for commit in commits_items:
            start_date1 = start_date.replace(tzinfo=timezone.utc)
            end_date1 = end_date.replace(tzinfo=timezone.utc)
            published_at = datetime.fromisoformat(commit['commit']['committer']['date'].replace("Z", "+00:00"))
            if start_date1 <= published_at <= end_date1:
                commits_response.append({ 
                    "committed_date": commit['commit']['committer']['date'], 
                })
        print(f"END OF COMMITS FOR PAGE : {i}")
    print("commits:\n")














    '''
        1. Hit LSTM Microservice by passing issues_response as body
        2. LSTM Microservice will give a list of string containing image paths hosted on google cloud storage
        3. On recieving a valid response from LSTM Microservice, append the above json_response with the response from
            LSTM microservice
    '''
    created_at_body = {
        "issues": issues_reponse,
        "type": "created_at",
        "repo": repo_name.split("/")[1]
    }
    closed_at_body = {
        "issues": issues_reponse,
        "type": "closed_at",
        "repo": repo_name.split("/")[1]
    }
    

    ### BODY CODE FOR COMMITS
    # commits_body = {
    #     "pull": commit_response,
    #     "type": "commits",
    #     "repo": repo_name.split("/")[1]
    # }

    ### BODY CODE FOR PULLS
    pulls_body = {
        "pull": pull_response,
        "type": "pull_request",
        "repo": repo_name.split("/")[1]
    }

    ### BODY CODE FOR BRANCH
    branch_body = {
        "branch": branch_response,
        "type": "branch",
        "repo": repo_name.split("/")[1]
    }

    # ### BODY CODE FOR CONTRIBUTORS
    contributors_body = {
        "contributor": contributors_response,
        "type":"contributor",
        "repo": repo_name.split("/")[1]
    }

    # ### BODY CODE FOR RELEASES
    releases_body = {
        "release": releases_response,
        "type":"release",
        "repo": repo_name.split("/")[1]
    }

    # # ### BODY CODE FOR COMMITS
    commits_body = {
        "commits": commits_response,
        "type": "commits",
        "repo": repo_name.split("/")[1]
    }
    
    issues_body = {
        "issues": issues_reponse,
        "type": "issues",
        "repo": repo_name.split("/")[1]
    }



    # LSTM_API_URL = "http://127.0.0.1:8080/" 
    LSTM_API_URL = 'https://lstm-pred-forecast-128813847309.us-central1.run.app/'

    '''
    Trigger the LSTM microservice to forecasted the created issues
    The request body consists of created issues obtained from GitHub API in JSON format
    The response body consists of Google cloud storage path of the images generated by LSTM microservice
    '''
    
    # print("STARTED FORECAST")
    
    # created_at_response = requests.post(LSTM_URL + "api/forecast",
    #                                     json=created_at_body,
    #                                     headers={'content-type': 'application/json'})
    # print("ENDED FORECAST")
    # print(created_at_response.status_code)
    # if created_at_response.status_code != 200:
    #     print(f"Error {created_at_response.status_code}: {created_at_response.text}")
    #     return ({ 'error' : created_at_response.text})
    # else:
    #     print("SAFE")
    #     print(created_at_response.json())
    # print("====\n\n")
    '''
    Trigger the LSTM microservice to forecasted the closed issues
    The request body consists of closed issues obtained from GitHub API in JSON format
    The response body consists of Google cloud storage path of the images generated by LSTM microservice
    '''    
    # closed_at_response = requests.post(LSTM_API_URL,
    #                                    json=closed_at_body,
    #                                    headers={'content-type': 'application/json'})

    # print("DONE COMMITS")
    # commitsResponse = requests.post("https://lstm-forecast-lstm-live-128813847309.us-central1.run.app/"+"api/commits",
    #                                    json=commits_body,
    #                                    headers={'content-type': 'application/json'})
    # print("DONE COMMITS API CALL")
    # print(commitsResponse.json())

    # print("DONE PULLS LSTM")
    # pulls_response = requests.post(LSTM_URL+"api/pulls",
    #                                    json = pulls_body,
    #                                    headers={'content-type': 'application/json'})
    # print("DONE PULLS LSTM API CALL")
    
    ########################################################### PHOPHETS ###############################################################

    # print("PERFORM fbProphets PULL")
    # pulls_prophet_response = requests.post(LSTM_URL+"api/fbprophetpull",
    #                                    json = pulls_body,
    #                                    headers={'content-type': 'application/json'})
    # print("DONE fbProphets PULL")
    # print(pulls_prophet_response)


    # print("PERFORM fbProphets COMMIT")
    # commits_prophet_response = requests.post(LSTM_URL+'/api/prophetcommits',
    #                                    json=commits_body,
    #                                    headers={'content-type': 'application/json'})
    # print("DONE fbProphets COMMIT")
    # print(commits_prophet_response)

    ##########################################################################################################################################

    # print("Issues fbProphets created by")
    # issues_created_prophet_response = requests.post(LSTM_URL+'/api/fbprophetCtd',
    #                                    json=created_at_body,
    #                                    headers={'content-type': 'application/json'})
    # print("DONE fbProphets created by")
    # print(issues_created_prophet_response)


    # print("Issues fbProphets closed by")
    # issues_closed_prophet_response = requests.post(LSTM_URL+'/api/fbprophetIssuesCls',
    #                                    json=closed_at_body,
    #                                    headers={'content-type': 'application/json'})
    # print("DONE fbProphets closed by")
    # print(issues_closed_prophet_response)

    ############################################## PULL ALL MODELS ##############################################

    print("CALLED PULL ALL MODELS API")
    pulls_all_models_response = requests.post(LSTM_API_URL + "api/allPullsModel",
                                       json = pulls_body,
                                       headers={'content-type': 'application/json'})
    print("DONE - PULL ALL MODELS API")
    print(pulls_all_models_response.status_code)
    if pulls_all_models_response.status_code != 200:
        print(f"Error {pulls_all_models_response.status_code}")
    else:
        print("SAFE allPullsModel")
        print(pulls_all_models_response.json())
    print("====\n\n")


    ############################################## BRANCH ALL MODELS ##############################################

    print("Called branch all Models")
    branch_all_models_response = requests.post(LSTM_API_URL+'/api/allBranchesModel',
                                       json=branch_body,
                                       headers={'content-type': 'application/json'})
    
    print("DONE branch all Models")
    print(branch_all_models_response)

    # ######################################## CONTRIBUTIONS ALL MODELS ########################################

    print("Called Contributors all Models")
    contributors_all_models_response = requests.post(LSTM_API_URL+'/api/allContributorsModel',
                                       json=contributors_body,
                                       headers={'content-type': 'application/json'})
    
    print("DONE Contributors all Models")
    print(contributors_all_models_response)

    
  ######################################## RELEASES ALL MODELS ########################################

    print("Called Releases all Models")
    releases_all_models_response = requests.post(LSTM_API_URL+'/api/allReleasesModel',
                                       json=releases_body,
                                       headers={'content-type': 'application/json'})
    
    print("DONE Releases all Models")
    print(releases_all_models_response)


    # ####################################### COMMITS ALL MODELS ########################################
    
    print("Called Commits all Models")
    commits_all_models_response = requests.post(LSTM_API_URL+'/api/allCommitsModel',
                                       json=commits_body,
                                       headers={'content-type': 'application/json'})
    
    print("DONE Commits all Models")
    print(commits_all_models_response)    

    ####################################### ISSUES CREATED_AT ALL MODELS ########################################
    
    print("Called issued Created at all Models")
    issues_created_at_all_models_response = requests.post(LSTM_API_URL + 'api/allIssuesCreatedAt',
                                       json=issues_body,
                                       headers={'content-type': 'application/json'})
    
    print("DONE issued Created at all Models")
    print(issues_created_at_all_models_response) 

    # ####################################### ISSUES CLOSED_AT ALL MODELS ########################################
    
    print("Called issued closed at all Models")
    issues_closed_at_all_models_response = requests.post(LSTM_API_URL + 'api/allIssuesClosedAt',
                                       json=issues_body,
                                       headers={'content-type': 'application/json'})
    
    print("DONE issued closed at all Models")
    print(issues_closed_at_all_models_response) 

    #####################################################################################################################







    
    '''
    Create the final response that consists of:
        1. GitHub repository data obtained from GitHub API
        2. Google cloud image urls of created and closed issues obtained from LSTM microservice
    '''
    json_response = {
        "created": created_at_issues,
        "closed": closed_at_issues,
        "starCount": repository["stargazers_count"],
        "forkCount": repository["forks_count"],
        # "createdAtImageUrls": {
        #     **created_at_response.json(),
        # },
        # "closedAtImageUrls": {
        #     **closed_at_response.json(),
        # },
        # "commit_response_lstm":{
        #     **commits_response.json(),
        # },
        # "pulls_response_lstm":{
        #     **pulls_response.json(),
        # },
        "pulls_all_models_response": {
            **pulls_all_models_response.json()
        },
        # "issues_created_prophet_response": {
        #     **issues_created_prophet_response.json()
        # },
        # "issues_closed_prophet_response": {
        #     **issues_closed_prophet_response.json()
        # },
        "branch_all_models_response": {
            **branch_all_models_response.json()
        },
        "contributors_all_models_response": {
            **contributors_all_models_response.json()
        },
        "releases_all_models_response": {
            **releases_all_models_response.json()
        },
        "commits_all_models_response": {
            **commits_all_models_response.json()
        },
        "issues_created_at_all_models_response": {
            **issues_created_at_all_models_response.json()
        },
        "issues_closed_at_all_models_response": {
            **issues_closed_at_all_models_response.json()
        }
    }
    # Return the response back to client (React app)
    return jsonify(json_response)


# Run flask app server on port 5000
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
