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

    $('#send-btn').click(sendMessage);
    $('#chat-input').keypress(function(e) {
        if (e.which === 13) sendMessage();
    });

    function sendMessage() {
        let message = $('#chat-input').val();
        if (message.trim() === '') return;

        // 使用者訊息
        $('#chat-messages').append(`
            <div class="chat-message user">
                <div class="chat-bubble user">${message}</div>
            </div>
        `);
        $('#chat-input').val('');

        $.ajax({
            url: '/chat',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ message: message }),
            success: function(data) {
                const response = data.response;
                const rooms = response.rooms;
                const conclusion = response.conclusion;

                // AI 結論訊息
                $('#chat-messages').append(`
                    <div class="chat-message ai">
                        <img src="/static/ai-avatar.png" class="chat-avatar">
                        <div>
                            <div class="chat-bubble ai">${conclusion.replace(/\n/g, '<br>')}</div>
                        </div>
                    </div>
                `);

                // 在對話流中插入每個卡片
                if (rooms && rooms.length > 0) {
                    rooms.forEach(room => {
                        const cardHtml = `
                            <div class="chat-message ai">
                                <img src="/static/ai-avatar.png" class="chat-avatar">
                                <div class="card mb-2" style="width: 100%;">
                                    <div class="card-body">
                                        <h5 class="card-title">${room.name}</h5>
                                        <h6 class="card-subtitle mb-2 text-muted">價格：${room.price} 元</h6>
                                        <p class="card-text">
                                            面積：${room.area}<br>
                                            特色：${room.features}
                                        </p>
                                    </div>
                                </div>
                            </div>
                        `;
                        $('#chat-messages').append(cardHtml);
                    });
                }

                $('#chat-messages').scrollTop($('#chat-messages')[0].scrollHeight);
            },
            error: function() {
                $('#chat-messages').append(`
                    <div class="chat-message ai">
                        <img src="/static/ai-avatar.png" class="chat-avatar">
                        <div class="chat-bubble ai">抱歉，伺服器出現問題，請稍後再試。</div>
                    </div>
                `);
            }
        });
    }
});
