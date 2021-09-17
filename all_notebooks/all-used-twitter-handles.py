import json
import csv


def convert_to_json():
    tweets = []
    with open("../input/tweets.csv", "r") as csv_file:

        reader = csv.reader(csv_file)
        rows = iter(reader)
        next(rows)
        for row in rows:
            tweet_object = dict(
                name=row[0],
                username=row[1],
                description=row[2],
                location=row[3],
                followers=row[4],
                numberstatuses=row[5],
                time=row[6],
                tweets=row[7]
            )
            tweets.append(tweet_object)
    
    return json.dumps(tweets, indent=4)


json_tweets = convert_to_json()
tweets = json.loads(json_tweets)
usernames = []
for tweet in tweets:
    username_string = "@" + tweet["username"]
    if username_string not in usernames:
        usernames.append(username_string)
    words = tweet["tweets"].split(" ")
    for word in words:
        if len(word) > 0 and ord(word[0]) is 64:
            embedded_user = word.split(":")[0]
            if embedded_user not in usernames:
                usernames.append(embedded_user)
print(len(usernames))

    # file = open("all_usernames.txt", "w")
    # usernames = sorted(usernames)
    # for username in usernames:
    #     file.write(username + "\n")
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.