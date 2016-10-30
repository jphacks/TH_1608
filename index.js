var request = require('request');
var Botkit = require('botkit');
var CronJob = require('cron').CronJob;

var controller = Botkit.slackbot();
var bot = controller.spawn({
    token: process.env.token
});

var options = {
    url: 'http://www.cl.ecei.tohoku.ac.jp/~ryo-t/wakarepo/api.py/',
    json: true,
    qs: {
        'screen_name': 'favofavo9'
    }
};

var flag = false;

bot.startRTM(function(err, bot, payload) {
    if (err) {
        throw new Error('Could not connect to Slack');
    }
    new CronJob({
        cronTime: '*/10 * * * * *',
        onTick: wakarepo,
        start: true,
        timeZone: 'Asia/Tokyo'
    });
});

controller.hears(['test'], ['direct_message', 'direct_mention', 'mention'], function(bot, message) {
    bot.reply(message, {
        text: 'test',
        username: 'わかれぽ'
    });
});

function wakarepo() {
    if (!flag) {
        request.get(options, function(error, response, body) {
            var user = 'favofavo9';
            if (!error && response.statusCode == 200) {
                if (body['broken_up'][user]) {
                    flag = true;
                    var text = '最近 ' + user + ' さんが別れました！';
                    bot.say({
                        channel: 'wakarepo',
                        text: text,
                        username: 'わかれぽ'
                    });
                } else {
                    var text = user + ' さんは *' + String(parseInt(Number(body['probs'][user]) * 100)) + '%* の確率で付き合っています';
                    bot.say({
                        channel: 'wakarepo',
                        text: text,
                        username: 'わかれぽ'
                    });
                }
            } else {
                console.log('error: ' + response.statusCode);
            }
        })
    }
}
