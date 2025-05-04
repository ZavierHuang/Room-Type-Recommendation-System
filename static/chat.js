$(document).ready(function() {
    // 收合側欄
    $('#chat-header').click(function() {
        $('#chat-sidebar').addClass('collapsed');
        $('#chat-toggle-btn').show();
        $('.container').removeClass('with-chat');
    });

    // 展開側欄
    $('#chat-toggle-btn').click(function() {
        $('#chat-sidebar').removeClass('collapsed');
        $('#chat-toggle-btn').hide();
        $('.container').addClass('with-chat');
    });

    // 送出按鈕
    $('#send-btn').click(function() {
        sendMessage();
    });

    // Enter 送出
    $('#chat-input').keypress(function(e) {
        if (e.which === 13) {
            sendMessage();
        }
    });

    function sendMessage() {
        let message = $('#chat-input').val();
        if (message.trim() === '') return;

        // 顯示使用者訊息
        $('#chat-messages').append(`<div><strong>你:</strong> ${message}</div>`);
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

                // 清空舊卡片
                $('#chat-cards').empty();

                // 如果有房型，顯示卡片
                if (rooms && rooms.length > 0) {
                    rooms.forEach(room => {
                        const cardHtml = `
                            <div class="card me-2" style="min-width: 250px;">
                                <div class="card-body">
                                    <h5 class="card-title">${room.name}</h5>
                                    <h6 class="card-subtitle mb-2 text-muted">價格：${room.price} 元</h6>
                                    <p class="card-text">
                                        面積：${room.area}<br>
                                        特色：${room.features}
                                    </p>
                                </div>
                            </div>
                        `;
                        $('#chat-cards').append(cardHtml);
                    });
                }

                // 顯示 AI 結論
                $('#chat-messages').append(`<div><strong>AI:</strong> ${conclusion.replace(/\n/g, '<br>')}</div>`);

                // 自動捲到最底
                $('#chat-messages').scrollTop($('#chat-messages')[0].scrollHeight);
            },
            error: function() {
                $('#chat-messages').append(`<div><strong>AI:</strong> 抱歉，伺服器出現問題，請稍後再試。</div>`);
                $('#chat-messages').scrollTop($('#chat-messages')[0].scrollHeight);
            }
        });
    }
});
