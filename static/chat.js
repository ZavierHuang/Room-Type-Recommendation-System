$(document).ready(function() {
    $('#chat-header').click(function() {
        $('#chat-sidebar').addClass('collapsed');
        $('#chat-toggle-btn').show();
        $('.container').removeClass('with-chat');
    });

    $('#chat-toggle-btn').click(function() {
        $('#chat-sidebar').removeClass('collapsed');
        $('#chat-toggle-btn').hide();
        $('.container').addClass('with-chat');
    });

    $('#send-btn').click(function() {
        sendMessage();
    });

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
            success: function() {
                $('#chat-messages').scrollTop($('#chat-messages')[0].scrollHeight);
            }
        });
    }
});
