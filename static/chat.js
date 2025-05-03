$(document).ready(function() {
    $('#toggle-chat').click(function() {
        $('#chatbox').toggle();
    });

    $('#send-btn').click(function() {
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
    });
});
