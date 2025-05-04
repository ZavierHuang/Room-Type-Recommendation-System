$(document).ready(function() {
    // 展開聊天室
    $('#chat-toggle-btn').click(function() {
        $('#chat-sidebar').removeClass('collapsed');
        $('#chat-toggle-btn').hide();
        $('.container').addClass('with-chat'); // 加上 margin
    });

    // 收合聊天室
    $('#chat-header').click(function() {
        $('#chat-sidebar').addClass('collapsed');
        $('#chat-toggle-btn').show();
        $('.container').removeClass('with-chat'); // 移除 margin
    });

    // 發送訊息
    $('#send-btn').click(function() {
        sendMessage();
    });

    // 按 Enter 也能送出
    $('#chat-input').keypress(function(e) {
        if (e.which === 13) {
            sendMessage();
        }
    });

    function sendMessage() {
        let message = $('#chat-input').val();
        if (message.trim() === '') return;

        $('#chat-messages').append(`<div><strong>你:</strong> ${message}</div>`);
        $('#chat-input').val('');

        $.ajax({
            url: '/chat',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ message: message }),
            success: function(data) {
                $('#chat-messages').append(`<div><strong>AI:</strong> ${data.response.replace(/\n/g, '<br>')}</div>`);
                $('#chat-messages').scrollTop($('#chat-messages')[0].scrollHeight);
            }
        });
    }
});
