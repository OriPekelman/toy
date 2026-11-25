# lib/toy/serve/openai/json_prefix.rb — is this text the start of a JSON
# object? (toy#219, `response_format: {"type": "json_object"}`.)
#
# JsonPrefix reads a text byte by byte and says whether it can still be
# continued to one JSON object (RFC 8259, the top-level value being an
# object). The serve decode step uses it to pick, among the model's most
# likely tokens, the first one the reply can continue with.
#
# Plain Ruby with no dependencies: the compiled server and the gates
# (under CRuby) run the same file.
#
# Whitespace between tokens is JSON, so a model could emit it for ever;
# more than WS_MAX whitespace bytes in a row are refused.

class JsonPrefix
  WS_MAX = 16

  # :stack  open containers, one byte each: "o" object, "a" array
  # :st     what may come next outside a string, number or literal:
  #           0 a value            1 a key or "}" (after "{")
  #           2 a key (after ",")  3 ":"
  #           4 "," or a close     5 a value or "]" (after "[")
  #           6 nothing: the object is closed
  #           7 "{" (nothing read yet)
  # :str    0 not in a string, 1 in a key, 2 in a string value
  # :esc    in a string: 0 plain, 1 after a backslash, 2 + n when n hex
  #         digits of a \uXXXX are still to come
  # :num    0 not in a number; 1 after "-", 2 after a leading "0", 3 integer
  #         digits, 4 after ".", 5 fraction digits, 6 after "e", 7 after the
  #         exponent's sign, 8 exponent digits
  # :lit    what is left of a true / false / null being read
  # :ws     whitespace bytes read in a row
  attr_accessor :stack, :st, :str, :esc, :num, :lit, :ws

  def initialize
    @stack = ""
    @st    = 7
    @str   = 0
    @esc   = 0
    @num   = 0
    @lit   = ""
    @ws    = 0
  end

  def copy_from(o)
    @stack = o.stack
    @st    = o.st
    @str   = o.str
    @esc   = o.esc
    @num   = o.num
    @lit   = o.lit
    @ws    = o.ws
    0
  end

  # The object is closed: nothing but whitespace may follow.
  def complete
    @st == 6
  end

  # Reads `s`. False when the text read so far, with `s`, is not the start
  # of a JSON object; the state is then undefined (feed a copy to try).
  def feed(s)
    n = s.bytesize
    i = 0
    while i < n
      if !feed_byte(s.getbyte(i))
        return false
      end
      i = i + 1
    end
    true
  end

  def value_done
    if @stack.bytesize == 0
      @st = 6
    else
      @st = 4
    end
    0
  end

  def push(kind)
    @stack = @stack + kind
    0
  end

  def top
    if @stack.bytesize == 0
      return 0
    end
    @stack.getbyte(@stack.bytesize - 1)
  end

  def pop
    @stack = @stack.byteslice(0, @stack.bytesize - 1)
    value_done
    0
  end

  def digit?(b)
    b >= 48 && b <= 57
  end

  def feed_byte(b)
    if @str != 0
      return feed_str(b)
    end
    if @lit.bytesize > 0
      if b != @lit.getbyte(0)
        return false
      end
      @lit = @lit.byteslice(1, @lit.bytesize - 1)
      if @lit.bytesize == 0
        value_done
      end
      return true
    end
    if @num != 0
      r = feed_num(b)
      if r == 1
        return true
      end
      if r < 0
        return false
      end
      # The number ended before this byte, which is read below.
      @num = 0
      value_done
    end
    if b == 32 || b == 9 || b == 10 || b == 13
      @ws = @ws + 1
      return @ws <= WS_MAX
    end
    @ws = 0
    if @st == 7
      if b == 123
        push("o")
        @st = 1
        return true
      end
      return false
    end
    if @st == 0 || @st == 5
      if @st == 5 && b == 93
        pop
        return true
      end
      return feed_value(b)
    end
    if @st == 1 || @st == 2
      if b == 34
        @str = 1
        return true
      end
      if @st == 1 && b == 125
        pop
        return true
      end
      return false
    end
    if @st == 3
      if b == 58
        @st = 0
        return true
      end
      return false
    end
    if @st == 4
      if b == 44
        if top == 111
          @st = 2
        else
          @st = 0
        end
        return true
      end
      if (b == 125 && top == 111) || (b == 93 && top == 97)
        pop
        return true
      end
      return false
    end
    false
  end

  # The first byte of a value.
  def feed_value(b)
    if b == 34
      @str = 2
    elsif b == 123
      push("o")
      @st = 1
    elsif b == 91
      push("a")
      @st = 5
    elsif b == 45
      @num = 1
    elsif b == 48
      @num = 2
    elsif b >= 49 && b <= 57
      @num = 3
    elsif b == 116
      @lit = "rue"
    elsif b == 102
      @lit = "alse"
    elsif b == 110
      @lit = "ull"
    else
      return false
    end
    true
  end

  # A byte inside a string.
  def feed_str(b)
    if @esc == 1
      if b == 117
        @esc = 6
        return true
      end
      if b == 34 || b == 92 || b == 47 || b == 98 || b == 102 || b == 110 || b == 114 || b == 116
        @esc = 0
        return true
      end
      return false
    end
    if @esc >= 3
      if digit?(b) || (b >= 65 && b <= 70) || (b >= 97 && b <= 102)
        @esc = @esc - 1
        if @esc == 2
          @esc = 0
        end
        return true
      end
      return false
    end
    if b == 92
      @esc = 1
      return true
    end
    if b == 34
      if @str == 1
        @st = 3
      else
        value_done
      end
      @str = 0
      return true
    end
    b >= 32
  end

  # A byte while a number is being read: 1 it is part of the number, 0 the
  # number ended before it, -1 the number cannot end or continue here.
  def feed_num(b)
    if @num == 1
      if b == 48
        @num = 2
        return 1
      end
      if b >= 49 && b <= 57
        @num = 3
        return 1
      end
      return -1
    end
    if @num == 4 || @num == 7
      if digit?(b)
        @num = @num + 1
        return 1
      end
      return -1
    end
    if @num == 6
      if b == 43 || b == 45
        @num = 7
        return 1
      end
      if digit?(b)
        @num = 8
        return 1
      end
      return -1
    end
    # 2, 3, 5, 8: the number could end here.
    if digit?(b)
      if @num == 2
        return -1
      end
      return 1
    end
    if b == 46 && (@num == 2 || @num == 3)
      @num = 4
      return 1
    end
    if (b == 101 || b == 69) && @num != 8
      @num = 6
      return 1
    end
    0
  end
end
