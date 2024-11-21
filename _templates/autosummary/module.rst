{{ fullname }}
{{ underline }}

{% for item in members %}
- **{{ item.name }}**: {{ item.summary }}
{% endfor %}
