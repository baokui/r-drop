from bert4keras.snippets import text_segmentate
def truncate(text,maxlen):
    """截断句子
    """
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    strips += '0123456789'
    strips += 'abcdefghijklmnopqrstuvwxyz'
    text = text.lower()
    return text_segmentate(text, maxlen - 2, seps, strips)[0]