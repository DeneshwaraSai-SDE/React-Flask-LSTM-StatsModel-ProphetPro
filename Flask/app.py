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
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import json
import dateutil.relativedelta
from datetime import date, datetime, timedelta, timezone
import pandas as pd
import aiohttp
import asyncio

app = Flask(__name__)
CORS(app)

async def async_get_request(session, url, headers=None, params=None):
    try:
        async with session.get(url, headers=headers, params=params) as response:
            return await response.json()
    except aiohttp.ClientError as e:
        print(f"GET request failed for {url}: {e}")
        return {"error": str(e)}

async def async_post_request(session, url, json_data, headers=None):
    try:
        async with session.post(url, json=json_data, headers=headers) as response:
            if response.status != 200:
                print(f"POST request failed for {url} with status {response.status}")
                return response.status, {"error": f"Status {response.status}"}
            try:
                return response.status, await response.json()
            except aiohttp.ContentTypeError as e:
                print(f"ContentTypeError for {url}: {e}")
                return response.status, {"error": "Invalid content type"}
    except aiohttp.ClientError as e:
        print(f"POST request failed for {url}: {e}")
        return 500, {"error": str(e)}

@app.route('/api/github', methods=['POST'])
async def github():
    print("STARTED GITHUB API PROCESSING")
    body = request.get_json()
    repo_name = body['repository']
    print(repo_name)
    
    
    GITHUB_TOKEN = ''
    token = os.environ.get('GITHUB_TOKEN', GITHUB_TOKEN)

    GITHUB_URL = "https://api.github.com/"
    headers = {"Authorization": f'token {token}', "Accept": "application/json"}
    params = {"state": "open"}

    async with aiohttp.ClientSession() as session:
        # Fetch repository details
        repository_url = GITHUB_URL + "repos/" + repo_name
        repository = await async_get_request(session, repository_url, headers=headers)
        print("repository: ",  repository)
        if "error" in repository:
            return Response(json.dumps(repository), mimetype='application/json', status=500)

        today = date.today()
        issues_response = []

        # Iterating to get issues for every month for the past 12 months
        for i in range(15):
            per_page = 'per_page=100'
            page = 'page='
            search_query = repo_name + '/issues?state=all' + "&" + per_page+ "&" + page + f'{i}'
            query_url = GITHUB_URL + "repos/" + search_query
            print(query_url)

            search_issues = await async_get_request(session, query_url, headers=headers,)
            if "error" in search_issues:
                return Response(json.dumps(search_issues), mimetype='application/json', status=500)

            issues_items = search_issues 

            if issues_items is None:
                error = {"error": "Data Not Available"}
                return Response(json.dumps(error), mimetype='application/json', status=500)
            print(issues_items)
            for issue in issues_items:
                label_name = []
                data = {}
                current_issue = issue
                data['issue_number'] = current_issue["number"]
                data['created_at'] = current_issue["created_at"][0:10]
                data['closed_at'] = current_issue["closed_at"][0:10] if current_issue["closed_at"] else None
                for label in current_issue["labels"]:
                    label_name.append(label["name"])
                data['labels'] = label_name
                data['State'] = current_issue["state"]
                data['Author'] = current_issue["user"]["login"]
                issues_response.append(data)
            print(f"END OF ISSUES FOR PAGE : {i}")

        # Process issues with pandas (synchronous)
        df = pd.DataFrame(issues_response)

        # Daily Created Issues
        df_created_at = df.groupby(['created_at'], as_index=False).count()
        dataFrameCreated = df_created_at[['created_at', 'issue_number']]
        dataFrameCreated.columns = ['date', 'count']

        created_at = df['created_at']
        month_issue_created = pd.to_datetime(pd.Series(created_at), format='%Y-%m-%d')
        month_issue_created.index = month_issue_created.dt.to_period('m')
        month_issue_created = month_issue_created.groupby(level=0).size()
        month_issue_created = month_issue_created.reindex(
            pd.period_range(month_issue_created.index.min(), month_issue_created.index.max(), freq='m'),
            fill_value=0
        )
        month_issue_created_dict = month_issue_created.to_dict()
        created_at_issues = [[str(key), month_issue_created_dict[key]] for key in month_issue_created_dict.keys()]

        closed_at = df['closed_at'].sort_values(ascending=True)
        month_issue_closed = pd.to_datetime(pd.Series(closed_at), format='%Y-%m-%d')
        month_issue_closed.index = month_issue_closed.dt.to_period('m')
        month_issue_closed = month_issue_closed.groupby(level=0).size()
        month_issue_closed = month_issue_closed.reindex(
            pd.period_range(month_issue_closed.index.min(), month_issue_closed.index.max(), freq='m'),
            fill_value=0
        )
        month_issue_closed_dict = month_issue_closed.to_dict()
        closed_at_issues = [[str(key), month_issue_closed_dict[key]] for key in month_issue_closed_dict.keys()]

        print("DONE FETCHING ISSUES:\n")

        # CODE FOR PULL REQUESTS
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)
        pull_response = []

        headers = {"Authorization": f'token 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', "Accept": "application/json"}

        for i in range(6):
            per_page = 'per_page=100'
            page = f'page={i}'
            search_query = repo_name + '/pulls?state=all' + "&" + per_page + "&" + page
            query_url = GITHUB_URL + "repos/" + search_query
            print(query_url)
            pull_requests = await async_get_request(session, query_url, headers=headers, params=params)
            if "error" in pull_requests:
                print(f"Pull request fetch failed: {pull_requests['error']}")
                continue

            if not pull_requests:
                continue
            else:
                print(f"{query_url} The page length is {len(pull_requests)}")

            for pull_req in pull_requests:
                label_name = []
                data = {}
                current_pull_req = pull_req

                if current_pull_req["created_at"]:
                    created_at_date = datetime.strptime(current_pull_req["created_at"][0:10], "%Y-%m-%d")
                    max_date = datetime.now() - timedelta(days=365 * 2)

                    if created_at_date > max_date:
                        data['pull_req_number'] = current_pull_req["number"]
                        data['created_at'] = current_pull_req["created_at"][0:10]
                        data['closed_at'] = current_pull_req["closed_at"][0:10] if current_pull_req["closed_at"] else None
                        for label in current_pull_req["labels"]:
                            label_name.append(label["name"])
                        data['labels'] = label_name
                        data['State'] = current_pull_req["state"]
                        data['Author'] = current_pull_req["user"]["login"]
                        pull_response.append(data)
            print("END OF Part - ", i)

        # COMMITS RESPONSE
        commits_response = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)
        headers = {"Authorization": f'token {token}', "Accept": "application/json"}
        for i in range(12):
            per_page = 'per_page=100'
            page = f'page={i}'
            search_query = repo_name + '/commits?state=all' + "&" + per_page + "&" + page
            query_url = GITHUB_URL + "repos/" + search_query
            commits_requests = await async_get_request(session, query_url, headers=headers, params=params)
            
            if "error" in commits_requests:
                print(f"Commits request fetch failed: {commits_requests['error']}")
                continue

            if not commits_requests or len(commits_requests) == 0:
                break
            else:
                print(f"{query_url} The page length is {len(commits_requests)}")

            for commit in commits_requests:
                start_date1 = start_date.replace(tzinfo=timezone.utc)
                end_date1 = end_date.replace(tzinfo=timezone.utc)
                published_at = datetime.fromisoformat(commit['commit']['committer']['date'].replace("Z", "+00:00"))
                if start_date1 <= published_at <= end_date1:
                    commits_response.append({
                        "committed_date": commit['commit']['committer']['date'],
                    })
            print(f"END OF COMMITS FOR PAGE : {i}")
        print("commits:\n")

        # BRANCH RESPONSE
        branch_response = []
        for i in range(12):
            per_page = 'per_page=100'
            page = f'page={i}'
            search_query = repo_name + '/branches?state=all' + "&" + per_page + "&" + page
            query_url = GITHUB_URL + "repos/" + search_query
            branch_requests = await async_get_request(session, query_url, headers=headers, params=params)
            if "error" in branch_requests:
                print(f"Branch request fetch failed: {branch_requests['error']}")
                continue
            if not branch_requests:
                continue
            else:
                print(f"LENGTH OF THIS PAGE : {len(branch_requests)}")

            for branch_req in branch_requests:
                current_branch_req = branch_req
                if not current_branch_req.get('commit', {}).get('sha'):
                    print("NO data found")
                    continue
                branch_commit_url = current_branch_req['commit']['url']
                commitedDetails = await async_get_request(session, branch_commit_url, headers=headers)
                if "error" in commitedDetails:
                    print(f"Commit details fetch failed: {commitedDetails['error']}")
                    continue
                commitDate = datetime.strptime(commitedDetails['commit']['author']['date'], "%Y-%m-%dT%H:%M:%SZ")
                max_date = datetime.now() - timedelta(days=730)
                if commitDate > max_date:
                    branch_response.append({'commit_date': commitedDetails['commit']['author']['date']})
            print(f"END OF BRANCHES FOR PAGE : {i}")
        print("branches:\n")

        # CONTRIBUTORS RESPONSE
        contributors_response = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)
        headers = {"Authorization": f'token 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', "Accept": "application/json"}
        for i in range(6):
            per_page = 'per_page=100'
            page = f'page={i}'
            search_query = repo_name + '/contributors?state=all' + "&" + per_page + "&" + page
            query_url = GITHUB_URL + "repos/" + search_query
            contributors_requests = await async_get_request(session, query_url, headers=headers, params=params)
            if "error" in contributors_requests:
                print(f"Contributors request fetch failed: {contributors_requests['error']}")
                continue

            if not contributors_requests:
                continue
            else:
                print(f"LENGTH OF THIS PAGE : {len(contributors_requests)}")

            for contributors_req in contributors_requests:
                login = contributors_req['login']
                userURL = f"https://api.github.com/users/{login}/events/public"
                eventRes = await async_get_request(session, userURL, headers=headers, params={"per_page": 100})
                if "error" in eventRes:
                    print(f"User events fetch failed for {login}: {eventRes['error']}")
                    continue
                else:
                    print(f"The lenght of the contrib: {len(eventRes)}")
                for event in eventRes:
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

        headers = {"Authorization": f'token {token}', "Accept": "application/json"}
        # RELEASE RESPONSE
        releases_response = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)

        for i in range(12):
            per_page = 'per_page=100'
            page = f'page={i}'
            search_query = repo_name + '/releases?state=all' + "&" + per_page + "&" + page
            query_url = GITHUB_URL + "repos/" + search_query
            releases_requests = await async_get_request(session, query_url, headers=headers, params=params)
            if "error" in releases_requests:
                print(f"Releases request fetch failed: {releases_requests['error']}")
                continue

            if not releases_requests or len(releases_requests) == 0:
                break
            else:
                print(f"LENGTH OF THIS PAGE : {len(releases_requests)}")

            for release in releases_requests:
                start_date1 = start_date.replace(tzinfo=timezone.utc)
                end_date1 = end_date.replace(tzinfo=timezone.utc)
                published_at = datetime.fromisoformat(release['published_at'].replace("Z", "+00:00"))
                if start_date1 <= published_at <= end_date1:
                    releases_response.append({
                        "published_at": release['published_at'],
                    })
            print(f"END OF RELEASES FOR PAGE : {i}")
        print("releases:\n")

        print('issues_response : ', len(issues_response))
        print('pull_response : ', len(pull_response))
        print('commits_response : ', len(commits_response))
        print('branch_response : ', len(branch_response))
        print('contributors_response : ', len(contributors_response ))
        print('releases_response : ', len(releases_response ))

        # Prepare request bodies
        issues_body = {
            "issues": issues_response,
            "type": "issues",
            "repo": repo_name.split("/")[1]
        }
        pulls_body = {
            "pull": pull_response,
            "type": "pull_request",
            "repo": repo_name.split("/")[1]
        }
        branch_body = {
            "branch": branch_response,
            "type": "branch",
            "repo": repo_name.split("/")[1]
        }
        contributors_body = {
            "contributor": contributors_response,
            "type": "contributor",
            "repo": repo_name.split("/")[1]
        }
        releases_body = {
            "release": releases_response,
            "type": "release",
            "repo": repo_name.split("/")[1]
        }
        commits_body = {
            "commits": commits_response,
            "type": "commits",
            "repo": repo_name.split("/")[1]
        }

        LSTM_API_URL = 'https://lstm-forecast-pred-128813847309.us-central1.run.app/'

        # Make async POST requests to LSTM microservice
        tasks = [
            async_post_request(session, LSTM_API_URL + "api/allPullsModel", pulls_body, headers={'content-type': 'application/json'}),
            async_post_request(session, LSTM_API_URL + "api/allBranchesModel", branch_body, headers={'content-type': 'application/json'}),
            async_post_request(session, LSTM_API_URL + "api/allContributorsModel", contributors_body, headers={'content-type': 'application/json'}),
            async_post_request(session, LSTM_API_URL + "api/allReleasesModel", releases_body, headers={'content-type': 'application/json'}),
            async_post_request(session, LSTM_API_URL + "api/allCommitsModel", commits_body, headers={'content-type': 'application/json'}),
            async_post_request(session, LSTM_API_URL + "api/allIssuesCreatedAt", issues_body, headers={'content-type': 'application/json'}),
            async_post_request(session, LSTM_API_URL + "api/allIssuesClosedAt", issues_body, headers={'content-type': 'application/json'}),
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process responses
        response_dicts = {
            'pulls_all_models_response': {},
            'branch_all_models_response': {},
            'contributors_all_models_response': {},
            'releases_all_models_response': {},
            'commits_all_models_response': {},
            'issues_created_at_all_models_response': {},
            'issues_closed_at_all_models_response': {}
        }
        endpoints = [
            'pulls', 'branch', 'contributors', 'releases', 'commits',
            'issues_created_at', 'issues_closed_at'
        ]

        for idx, response in enumerate(responses):
            endpoint = endpoints[idx]
            response_key = list(response_dicts.keys())[idx]

            if isinstance(response, Exception):
                print(f"Exception in {endpoint} request: {response}")
                continue

            status, data = response
            if status != 200 or "error" in data:
                print(f"Error in {endpoint} request: Status {status}, Data {data}")
            else:
                print(f"SAFE {endpoint}")
                response_dicts[response_key] = data

        # Create the final response
        json_response = {
            "created": created_at_issues,
            "closed": closed_at_issues,
            "starCount": repository.get("stargazers_count", 0),
            "forkCount": repository.get("forks_count", 0),
            "pulls_all_models_response": response_dicts['pulls_all_models_response'],
            "branch_all_models_response": response_dicts['branch_all_models_response'],
            "contributors_all_models_response": response_dicts['contributors_all_models_response'],
            "releases_all_models_response": response_dicts['releases_all_models_response'],
            "commits_all_models_response": response_dicts['commits_all_models_response'],
            "issues_created_at_all_models_response": response_dicts['issues_created_at_all_models_response'],
            "issues_closed_at_all_models_response": response_dicts['issues_closed_at_all_models_response']
        }

        return jsonify(json_response)

# Run flask app server on port 5000
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
